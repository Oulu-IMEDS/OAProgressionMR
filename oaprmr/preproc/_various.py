import torch


class NumpyToTensor(object):
    def __call__(self, *args):
        if len(args) > 1:
            return [torch.from_numpy(e.copy()) for e in args]
        else:
            return torch.from_numpy(args[0].copy())


class TensorToNumpy(object):
    def __call__(self, *args):
        if len(args) > 1:
            return [e.numpy() for e in args]
        else:
            return args[0].numpy()


class TensorToDevice(object):
    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, *args):
        return [e.to(self.device) for e in args]
