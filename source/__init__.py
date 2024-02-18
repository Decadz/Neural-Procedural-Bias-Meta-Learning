# Importing the training functions.
from source.meta_learning_default import meta_training_default
from source.meta_learning_default import meta_testing_default
from source.meta_learning_npbml import meta_training_npbml
from source.meta_learning_npbml import meta_testing_npbml
from source.meta_learning_relation import meta_training_relation
from source.meta_learning_relation import meta_testing_relation
from source.pretraining import pretrain

# Importing the default base-networks.
from source.models.convnet import Conv32, Conv48, Conv64, Conv128
from source.models.resnet import ResNet, WideResNet

# Importing the custom base-networks.
from source.models.adaconvnet import AdaConv32, AdaConv48, AdaConv64, AdaConv128
from source.models.adaresnet import AdaResNet, WideAdaResNet
from source.models.adalossnet import AdaLossNetwork

# Importing the relation network.
from source.models.relationnet import RelationNetwork, WideRelationNetwork

model_archive = {
    "conv32": Conv32,
    "conv48": Conv48,
    "conv64": Conv64,
    "conv128": Conv128,

    "resnet": ResNet,
    "wideresnet": WideResNet,

    "adaconv32": AdaConv32,
    "adaconv48": AdaConv48,
    "adaconv64": AdaConv64,
    "adaconv128": AdaConv128,

    "adaresnet": AdaResNet,
    "wideadaresnet": WideAdaResNet,

    "relationnet": RelationNetwork,
    "widerelationnet": WideRelationNetwork
}
