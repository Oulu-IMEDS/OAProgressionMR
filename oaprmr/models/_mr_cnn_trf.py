import copy
from collections import OrderedDict
import torch
from torch import nn
from einops import rearrange, repeat

from ._core_trf import FeaT
from ._core_fes import dict_fes


class MRCnnTrf(nn.Module):
    def __init__(self, config, path_weights):
        super(MRCnnTrf, self).__init__()
        self.config = config
        if self.config["debug"]:
            print("Config at model init", self.config)
        self.vs = dict()

        t_fe = dict_fes[self.config["fe"]["arch"]](
            pretrained=self.config["fe"]["pretrained"])
        if self.config["fe"]["with_gap"]:
            # Exclude trailing FC layer
            t_fe = list(t_fe.children())[:-1]
        else:
            # Exclude trailing GAP and FC layers
            t_fe = list(t_fe.children())[:-2]
        self._fe = nn.Sequential(*t_fe)

        if self.config["fe"]["dropout"]:
            self._fe_drop = nn.Dropout2d(p=self.config["fe"]["dropout"])
        else:
            self._fe_drop = nn.Identity()

        if self.config["debug"]:
            print("FE submodel", self._fe)

        if self.config["fe"]["arch"] in ("resnet18", "resnet34"):
            self.vs["fe_out_ch"] = 512
        elif self.config["fe"]["arch"] == "resnet50":
            self.vs["fe_out_ch"] = 2048
        else:
            raise ValueError(f"Unsupported `model.fe.arch`")

        # Calculate input data shape
        t = self.config["input_size"][0]
        if self.config["downscale"]:
            t = [round(s * d) for s, d in zip(t, self.config["downscale"][0])]
        self.vs["shape_in"] = t

        if self.config["fe"]["with_gap"]:
            self.vs["fe_out_spat"] = (1, 1, 1)
        else:
            try:
                mapping = {320: 10, 160: 5, 128: 4, 96: 3, 64: 2, 32: 1}
                self.vs["fe_out_spat"] = tuple(mapping[e] for e in self.vs["shape_in"])
            except (ValueError, IndexError) as e:
                msg = "Unspecified `model.fe` output shape for given `model.input_size`"
                raise ValueError(msg)

        if self.config["fe"]["dims_view"] == "rc":
            self.vs["agg_in_len"] = self.vs["shape_in"][2] * \
                                    (self.vs["fe_out_spat"][0] * self.vs["fe_out_spat"][1])
        elif self.config["fe"]["dims_view"] == "cs":
            self.vs["agg_in_len"] = self.vs["shape_in"][0] * \
                                    (self.vs["fe_out_spat"][1] * self.vs["fe_out_spat"][2])
        elif self.config["fe"]["dims_view"] == "rs":
            self.vs["agg_in_len"] = self.vs["shape_in"][1] * \
                                    (self.vs["fe_out_spat"][0] * self.vs["fe_out_spat"][2])
        else:
            raise ValueError(f"Unsupported `model.fe.dims_view`")

        self.vs["agg_in_depth"] = self.vs["fe_out_ch"]

        self._agg = FeaT(
            num_patches=self.vs["agg_in_len"],
            patch_dim=self.vs["agg_in_depth"],
            # emb_dim=self.config["agg"]["emb_dim"],
            emb_dim=self.vs["agg_in_depth"],
            depth=self.config["agg"]["depth"],
            heads=self.config["agg"]["heads"],
            mlp_dim=self.config["agg"]["mlp_dim"],
            num_classes=self.config["output_channels"],
            emb_dropout=self.config["agg"]["emb_dropout"],
            # with_cls=True,
            # num_cls_tokens=1,
            mlp_dropout=self.config["agg"]["mlp_dropout"],
            # num_outputs=1,
        )

        if self.config["restore_weights"]:
            self.load_state_dict(torch.load(path_weights))

    def _debug_tensor_shape(self, tensor, name=""):
        if self.config["debug"]:
            print(f"Shape of {name} is", tensor.size())

    def forward(self, input):
        """
        input : (B, CH, R, C, S)

        Notes:
            B - batch, CH - channel, R - row, C - column, S - slice/plane, F - feature
        """
        endpoints = OrderedDict()

        shapes = input.size()
        self._debug_tensor_shape(input, "input")

        t_in = repeat(input, "b ch r c s -> b (k ch) r c s", k=3)

        if self.config["fe"]["dims_view"] == "rc":
            t_in = rearrange(t_in, "b ch r c s -> (b s) ch r c")
        elif self.config["fe"]["dims_view"] == "cs":
            t_in = rearrange(t_in, "b ch r c s -> (b r) ch c s")
        elif self.config["fe"]["dims_view"] == "rs":
            t_in = rearrange(t_in, "b ch r c s -> (b c) ch r s")

        self._debug_tensor_shape(t_in, "proc in")

        res_fe = self._fe(t_in)
        self._debug_tensor_shape(res_fe, "FE out")
        t_fe = self._fe_drop(res_fe)
        t_fe = rearrange(t_fe, "(b d2) ch d0 d1 -> b (d2 d0 d1) ch", b=shapes[0])
        self._debug_tensor_shape(t_fe, "FE proc")

        res_agg, _, _ = self._agg(t_fe)
        self._debug_tensor_shape(res_agg, "AGG out")

        res_out = rearrange(res_agg, "b head cls -> b (head cls)")

        endpoints["main"] = res_out
        return endpoints


class MultiviewCnnTrf(nn.Module):
    def __init__(self, config, path_weights):
        super(MultiviewCnnTrf, self).__init__()
        self.config = config
        if self.config["debug"]:
            print("Config at model init", self.config)
        self.vs = dict()

        t_fe = dict_fes[self.config["fe"]["arch"]](
            pretrained=self.config["fe"]["pretrained"])
        if self.config["fe"]["with_gap"]:
            # Exclude trailing FC layer
            t_fe = list(t_fe.children())[:-1]
        else:
            # Exclude trailing GAP and FC layers
            t_fe = list(t_fe.children())[:-2]

        self._fes = nn.ModuleList()
        if self.config["fe"]["shared"]:
            self.vs["num_fe"] = 1
        else:
            self.vs["num_fe"] = 3
        for _ in range(self.vs["num_fe"]):
            self._fes.append(nn.Sequential(*copy.deepcopy(t_fe)))

        self._fe_drops = nn.ModuleList()
        for _ in range(self.vs["num_fe"]):
            if self.config["fe"]["dropout"]:
                self._fe_drops.append(nn.Dropout2d(p=self.config["fe"]["dropout"]))
            else:
                self._fe_drops.append(nn.Identity())

        if self.config["debug"]:
            print("FE submodel", self._fes)

        if self.config["fe"]["arch"] in ("resnet18", "resnet34"):
            self.vs["fe_out_ch"] = 512
        elif self.config["fe"]["arch"] == "resnet50":
            self.vs["fe_out_ch"] = 2048
        else:
            raise ValueError(f"Unsupported `model.fe.arch`")

        # Calculate input data shape
        t = self.config["input_size"][0]
        if self.config["downscale"]:
            t = [round(s * d) for s, d in zip(t, self.config["downscale"][0])]
        self.vs["shape_in"] = t

        if self.config["fe"]["with_gap"]:
            self.vs["fe_out_spat"] = (1, 1, 1)
        else:
            try:
                mapping = {320: 10, 160: 5, 128: 4, 96: 3, 64: 2, 32: 1}
                self.vs["fe_out_spat"] = tuple(mapping[e] for e in self.vs["shape_in"])
            except (ValueError, IndexError) as e:
                msg = "Unspecified `model.fe` output shape for given `model.input_size`"
                raise ValueError(msg)

        self.vs["patches_v0"] = self.vs["shape_in"][2] * \
                                (self.vs["fe_out_spat"][0] * self.vs["fe_out_spat"][1])
        self.vs["patches_v1"] = self.vs["shape_in"][0] * \
                                (self.vs["fe_out_spat"][1] * self.vs["fe_out_spat"][2])
        self.vs["patches_v2"] = self.vs["shape_in"][1] * \
                                (self.vs["fe_out_spat"][0] * self.vs["fe_out_spat"][2])
        self.vs["agg_in_len"] = self.vs["patches_v0"] + \
                                self.vs["patches_v1"] + \
                                self.vs["patches_v2"]

        self.vs["agg_in_depth"] = self.vs["fe_out_ch"]

        self._agg = FeaT(
            num_patches=self.vs["agg_in_len"],
            patch_dim=self.vs["agg_in_depth"],
            emb_dim=self.vs["agg_in_depth"],
            depth=self.config["agg"]["depth"],
            heads=self.config["agg"]["heads"],
            mlp_dim=self.config["agg"]["mlp_dim"],
            num_classes=self.config["output_channels"],
            emb_dropout=self.config["agg"]["emb_dropout"],
            # with_cls=True,
            # num_cls_tokens=1,
            mlp_dropout=self.config["agg"]["mlp_dropout"],
            # num_outputs=1,
        )

        if self.config["restore_weights"]:
            self.load_state_dict(torch.load(path_weights))

    def _debug_tensor_shape(self, tensor, name=""):
        if self.config["debug"]:
            print(f"Shape of {name} is", tensor.size())

    def forward(self, input):
        """
        input : (B, CH, R, C, S)

        Notes:
            B - batch, CH - channel, R - row, C - column, S - slice/plane, F - feature
        """
        endpoints = OrderedDict()

        shapes = input.size()
        self._debug_tensor_shape(input, "input")

        t_in = repeat(input, "b ch r c s -> b (k ch) r c s", k=3)
        t_in_v0 = rearrange(t_in, "b ch r c s -> (b s) ch r c")
        t_in_v1 = rearrange(t_in, "b ch r c s -> (b r) ch c s")
        t_in_v2 = rearrange(t_in, "b ch r c s -> (b c) ch r s")
        self._debug_tensor_shape(t_in, "proc in")
        self._debug_tensor_shape(t_in_v0, "proc V0 in")
        self._debug_tensor_shape(t_in_v1, "proc V1 in")
        self._debug_tensor_shape(t_in_v2, "proc V2 in")

        if self.config["fe"]["shared"]:
            res_fe_v0 = self._fes[0](t_in_v0)
            res_fe_v1 = self._fes[0](t_in_v1)
            res_fe_v2 = self._fes[0](t_in_v2)
        else:
            res_fe_v0 = self._fes[0](t_in_v0)
            res_fe_v1 = self._fes[1](t_in_v1)
            res_fe_v2 = self._fes[2](t_in_v2)
        self._debug_tensor_shape(res_fe_v0, "FE V0 out")
        self._debug_tensor_shape(res_fe_v1, "FE V1 out")
        self._debug_tensor_shape(res_fe_v2, "FE V2 out")

        if self.config["fe"]["shared"]:
            t_fe_v0 = self._fe_drops[0](res_fe_v0)
            t_fe_v1 = self._fe_drops[0](res_fe_v1)
            t_fe_v2 = self._fe_drops[0](res_fe_v2)
        else:
            t_fe_v0 = self._fe_drops[0](res_fe_v0)
            t_fe_v1 = self._fe_drops[1](res_fe_v1)
            t_fe_v2 = self._fe_drops[2](res_fe_v2)
        t_fe_v0 = rearrange(t_fe_v0, "(b s) ch d0 d1 -> b (s d0 d1) ch", b=shapes[0])
        t_fe_v1 = rearrange(t_fe_v1, "(b r) ch d0 d1 -> b (r d0 d1) ch", b=shapes[0])
        t_fe_v2 = rearrange(t_fe_v2, "(b c) ch d0 d1 -> b (c d0 d1) ch", b=shapes[0])
        self._debug_tensor_shape(t_fe_v0, "FE V0 proc")
        self._debug_tensor_shape(t_fe_v1, "FE V1 proc")
        self._debug_tensor_shape(t_fe_v2, "FE V2 proc")

        t_fe_vall = torch.cat([t_fe_v0, t_fe_v1, t_fe_v2], dim=1)
        self._debug_tensor_shape(t_fe_vall, "FE ALL out")
        res_agg, _, _ = self._agg(t_fe_vall)
        self._debug_tensor_shape(res_agg, "AGG out")

        res_out = rearrange(res_agg, "b head cls -> b (head cls)")

        endpoints["main"] = res_out
        return endpoints
