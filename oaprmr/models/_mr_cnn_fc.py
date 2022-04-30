import math
from collections import OrderedDict
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from ._core_fes import dict_fes


class MRCnnFc(nn.Module):
    def __init__(self, config, path_weights):
        super(MRCnnFc, self).__init__()
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

        self.vs["num_slices"] = self.config["input_size"][0][2]
        if self.config["downscale"]:
            self.vs["num_slices"] = round(self.vs["num_slices"] *
                                          self.config["downscale"][0][2])

        if self.config["fe"]["arch"] in ("resnet18", "resnet34"):
            self.vs["fe_out_ch"] = 512
        elif self.config["fe"]["arch"] == "resnet50":
            self.vs["fe_out_ch"] = 2048
        else:
            raise ValueError(f"Unsupported `model.fe.arch`")

        if self.config["fe"]["with_gap"]:
            self.vs["fe_out_spat"] = (1, 1)
        else:
            if self.config["input_size"][0][0] == 320:
                self.vs["fe_out_spat"] = (5, 5)
            else:
                msg = "Unspecified `model.fe` output shape for given `model.input_size`"
                raise ValueError(msg)

        if self.config["agg"]["kind"] == "concat":
            self._agg = nn.Sequential(*[
                Rearrange("(b s) ch d0 d1 -> b (s ch d0 d1)", s=self.vs["num_slices"]),
                nn.Linear(self.vs["num_slices"] *
                          self.vs["fe_out_ch"] *
                          math.prod(self.vs["fe_out_spat"]),
                          self.config["agg"]["hidden_size"]),
                nn.ReLU(),
                nn.Dropout(self.config["agg"]["dropout"]),
            ])

            self._agg_fc = nn.Linear(self.config["agg"]["hidden_size"],
                                     self.config["output_channels"])
        elif self.config["agg"]["kind"] == "avg_pool":
            self._agg = nn.Sequential(*[
                Rearrange("(b s) ch d0 d1 -> b s ch d0 d1", s=self.vs["num_slices"]),
                Reduce("b s ch d0 d1 -> b ch d0 d1", reduction="mean"),
                Rearrange("b ch d0 d1 -> b (ch d0 d1)"),
                nn.Dropout(self.config["agg"]["dropout"]),
            ])

            self._agg_fc = nn.Linear(self.vs["fe_out_ch"] *
                                     math.prod(self.vs["fe_out_spat"]),
                                     self.config["output_channels"])
        else:
            raise ValueError(f"Unsupported `model.agg.kind`")

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

        self._debug_tensor_shape(input, "input")

        tmp_in = rearrange(input, "b ch r c s -> (b s) ch r c")
        tmp_in = repeat(tmp_in, "bs ch r c -> bs (k ch) r c", k=3)
        self._debug_tensor_shape(tmp_in, "proc in")

        res_fe = self._fe(tmp_in)
        self._debug_tensor_shape(res_fe, "FE out")

        res_agg = self._agg(res_fe)
        self._debug_tensor_shape(res_agg, "AGG out")

        res_out = self._agg_fc(res_agg)
        self._debug_tensor_shape(res_out, "AGG FC out")

        endpoints["main"] = res_out
        return endpoints
