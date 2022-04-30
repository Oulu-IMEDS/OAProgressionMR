from ._core_trf import Transformer, FeaT, FeedForward, Attention
from ._xr_cnn import XRCnn
from ._mr_cnn_fc import MRCnnFc
from ._mr_cnn_lstm import MRCnnLstm
from ._mr_cnn_trf import MRCnnTrf, MultiviewCnnTrf
from ._resnet2p1d import ResNet2P1D50
from ._resnet_resnext_3d import ResNet3D50, ResNeXt3D50
from ._shufflenet3d import ShuffleNet3D


dict_models = {
    "XRCnn": XRCnn,
    "MRCnnFc": MRCnnFc,
    "MRCnnLstm": MRCnnLstm,
    "MRCnnTrf": MRCnnTrf,
    "MultiviewCnnTrf": MultiviewCnnTrf,
    "ResNet2P1D50": ResNet2P1D50,
    "ResNet3D50": ResNet3D50,
    "ResNeXt3D50": ResNeXt3D50,
    "ShuffleNet3D": ShuffleNet3D,
}
