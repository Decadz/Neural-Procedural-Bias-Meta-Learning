# Importing the training functions.
from source.meta_learning_default import meta_training_default
from source.meta_learning_default import meta_testing_default
from source.meta_learning_npbml import meta_training_npbml
from source.meta_learning_npbml import meta_testing_npbml
from source.pretraining import pretrain

# Importing the default base-networks.
from source.models.convnet import Conv, WideConv
from source.models.convnet import LinearWarpConv, LinearWarpWideConv
from source.models.convnet import WarpConv, WarpWideConv
from source.models.resnet import ResNet, WideResNet
from source.models.resnet import LinearWarpResNet, LinearWarpWideResNet
from source.models.resnet import WarpResNet, WarpWideResNet

# Importing the custom base-networks.
from source.models.adaconvnet import AdaConv32, AdaConv48, AdaConv64, AdaConv128
from source.models.adaconvnet import AdaWarpConv32, AdaWarpConv48, AdaWarpConv64, AdaWarpConv128
from source.models.adalossnet import AdaLossNetwork

model_archive = {
    "conv": Conv,
    "wideconv": WideConv,
    "linearwarpconv": LinearWarpConv,
    "linearwarpwideconv": LinearWarpWideConv,
    "warpconv": WarpConv,
    "warpwideconv": WarpWideConv,
    "resnet": ResNet,
    "wideresnet": WideResNet,
    "linearwarpresnet": LinearWarpResNet,
    "linearwarpwideresnet": LinearWarpWideResNet,
    "warpresnet": WarpResNet,
    "warpwideresnet": WarpWideResNet,

    "adaconv32": AdaConv32,
    "adaconv48": AdaConv48,
    "adaconv64": AdaConv64,
    "adaconv128": AdaConv128,
    "adawarpconv32": AdaWarpConv32,
    "adawarpconv48": AdaWarpConv48,
    "adawarpconv64": AdaWarpConv64,
    "adawarpconv128": AdaWarpConv128,
}
