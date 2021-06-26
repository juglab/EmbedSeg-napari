from embedseg_napari.models.BranchedERFNet import BranchedERFNet
from embedseg_napari.models.BranchedERFNet_3d import BranchedERFNet_3d
def get_model(name, model_opts):
    if name == "branched_erfnet":
        model = BranchedERFNet(**model_opts)
        return model
    elif name=="branched_erfnet_3d":
        model = BranchedERFNet_3d(**model_opts)
        return model
    else:
        raise RuntimeError("model \"{}\" not available".format(name))