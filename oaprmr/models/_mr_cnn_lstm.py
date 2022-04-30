import math
from collections import OrderedDict
import torch
from torch import nn
from einops import rearrange, repeat

from ._core_fes import dict_fes


class MRCnnLstm(nn.Module):
    def __init__(self, config, path_weights):
        super(MRCnnLstm, self).__init__()
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

        if self.config["fe"]["with_gap"]:
            self.vs["fe_out_spat"] = (1, 1)
        else:
            if self.config["input_size"][0][0] == 320:
                self.vs["fe_out_spat"] = (5, 5)
            else:
                msg = "Unspecified `model.fe` output shape for given `model.input_size`"
                raise ValueError(msg)

        self.vs["agg_in_depth"] = self.vs["fe_out_ch"]
        if self.config["agg"]["with_tokens"]:
            self.vs["agg_in_depth"] += 2

        self._agg = nn.LSTM(
            input_size=self.vs["agg_in_depth"],
            hidden_size=self.config["agg"]["hidden_size"],
            num_layers=self.config["agg"]["num_layers"],
            # bias=True,
            batch_first=True,
            dropout=self.config["agg"]["dropout"],
            bidirectional=True,
            # proj_size=None
        )
        self._agg_fc = nn.Linear(2 * self.config["agg"]["hidden_size"],
                                 self.config["output_channels"])

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

        t_in = rearrange(input, "b ch r c s -> (b s) ch r c")
        t_in = repeat(t_in, "bs ch r c -> bs (k ch) r c", k=3)
        self._debug_tensor_shape(t_in, "proc in")

        res_fe = self._fe(t_in)
        self._debug_tensor_shape(res_fe, "FE out")
        t_fe = rearrange(res_fe, "(b s) ch d0 d1 -> b (s d0 d1) ch", b=shapes[0])
        self._debug_tensor_shape(t_fe, "FE proc")

        if self.config["agg"]["with_tokens"]:
            num_slice = shapes[-1]
            num_superpix = math.prod(self.vs["fe_out_spat"])

            tok_slice = torch.linspace(start=0., end=1., steps=num_slice)
            tok_superpix = torch.linspace(start=0., end=1., steps=num_superpix)

            # Same priority as for data
            t = torch.meshgrid(tok_slice, tok_superpix)
            tok_comb = torch.stack([torch.flatten(t[0]),
                                    torch.flatten(t[1])], dim=1)
            tok_comb = repeat(tok_comb, "p t -> b p t", b=shapes[0])
            tok_comb = tok_comb.to(self.config["device"])

            t_fe = torch.cat([t_fe, tok_comb], dim=2)

        res_agg, _ = self._agg(t_fe)
        self._debug_tensor_shape(res_agg, "AGG out")

        res_agg = rearrange(res_agg, "b s (d f) -> b s d f", d=2)
        t_out = torch.cat([res_agg[:, -1, 0, :], res_agg[:, 0, 1, :]], dim=1)
        self._debug_tensor_shape(t_out, "AGG proc")

        res_out = self._agg_fc(t_out)
        self._debug_tensor_shape(res_out, "AGG FC out")

        endpoints["main"] = res_out
        return endpoints
