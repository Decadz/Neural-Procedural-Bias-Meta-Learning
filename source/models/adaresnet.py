import collections
import torch


class _AdaResNet(torch.nn.Module):

    def __init__(self, block_config, input_channels=3, num_ways=5, **kwargs):
        super(_AdaResNet, self).__init__()

        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("block1", _ConvBlock(input_channels, block_config[0])),
            ("block2", _ConvBlock(block_config[0], block_config[1])),
            ("block3", _ConvBlock(block_config[1], block_config[2])),
            ("adapt", _FiLMWarpConvBlock(block_config[2], block_config[3])),
            ("adaPool", torch.nn.AdaptiveAvgPool2d(1)),
            ("flatten", torch.nn.Flatten())
        ]))

        # Creating the permutation invariant head of the network.
        self.classifier = _PermutationInvariantClassifier(block_config[-1], num_ways)

        # Model configurations and hyper-parameters.
        self.input_channels = input_channels
        self.block_config = block_config
        self.num_ways = num_ways

        # Initializing the model's parameters.
        self.initialize()

    def forward(self, x, task_adaptive=False):

        # Turning on the task adaptive FiLM layers.
        for name, module in self.encoder.named_children():
            if isinstance(module, _FiLMWarpConvBlock):
                module.task_adaptive = task_adaptive

        # Generating the image embeddings using the encoder.
        z = self.encoder(x)

        # Generating the model predictions.
        return self.classifier(z)

    def initialize(self):
        # Initializing the networks parameters.
        self.classifier.initialize()
        for name, module in self.encoder.named_children():
            if isinstance(module, (_ConvBlock, _FiLMWarpConvBlock)):
                module.initialize()

    def reset_classifier(self):
        # Resetting the output layer using the output cone.
        self.classifier.reset_classifier()

    def meta_parameters(self):
        for name, module in self.encoder.named_children():
            if hasattr(module, "meta_parameters"):
                if isinstance(module, _FiLMWarpConvBlock):
                    yield from module.meta_parameters()
        yield from self.classifier.meta_parameters()

    def base_parameters(self):
        for name, module in self.encoder.named_children():
            if hasattr(module, "base_parameters"):
                if isinstance(module, _FiLMWarpConvBlock):
                    yield from module.base_parameters()
        yield from self.classifier.base_parameters()

    def pretraining_parameters(self):
        for name, module in self.encoder.named_children():
            if hasattr(module, "base_parameters"):
                yield from module.base_parameters()
        yield from self.classifier.base_parameters()


# ============================================================
# Network block definitions.
# ============================================================

class _ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock, self).__init__()

        # Defining the stacked convolutional block.
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)

        self.conv3 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)

        # Defining the residual skip connection layers.
        self.res_conv = torch.nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=False)
        self.res_bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)

        # Down-sampling layer at the end of the block.
        self.pool = torch.nn.MaxPool2d(2)

        # The activation function used in the network.
        self.activation = torch.nn.LeakyReLU(inplace=True)

    def forward(self, x):
        z = self.activation(self.bn1(self.conv1(x)))
        z = self.activation(self.bn2(self.conv2(z)))
        z = self.bn3(self.conv3(z)) + self.res_bn(self.res_conv(x))
        return self.pool(self.activation(z))

    def initialize(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight, 0, 0.01)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def base_parameters(self):
        yield from self.conv1.parameters()
        yield from self.conv2.parameters()
        yield from self.conv3.parameters()
        yield from self.res_conv.parameters()


class _FiLMWarpConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_FiLMWarpConvBlock, self).__init__()

        # Defining the stacked convolutional block.
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.film1 = torch.nn.Linear(in_channels, out_channels * 2)

        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.film2 = torch.nn.Linear(out_channels, out_channels * 2)

        self.conv3 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.film3 = torch.nn.Linear(out_channels, out_channels * 2)

        # Defining the warp preconditioning layers.
        self.warp_conv = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.warp_bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.warp_film = torch.nn.Linear(out_channels, out_channels * 2)

        # Defining the residual skip connection layers.
        self.res_conv = torch.nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=False)
        self.res_bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)

        # Down-sampling layer at the end of the block.
        self.pool = torch.nn.MaxPool2d(2)

        # The activation function used in the network.
        self.activation = torch.nn.LeakyReLU(inplace=True)

        # Field for controlling the current state of the layer.
        self.task_adaptive = False

    def forward(self, x):
        z = self.activation(self._apply_film(x, self.bn1(self.conv1(x)), self.film1))
        z = self.activation(self._apply_film(z, self.bn2(self.conv2(z)), self.film2))
        z = self._apply_film(z, self.bn3(self.conv3(z)), self.film3)
        z = self._apply_film(z, self.warp_bn(self.warp_conv(z)), self.warp_film)
        return self.pool(self.activation(z + self.res_bn(self.res_conv(x))))

    def _apply_film(self, x, z, film):

        if self.task_adaptive:  # If task adaptive apply Feature Wise Linear Modulation (FiLM).

            # Computing the local and global embeddings.
            avg_channel = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))

            # Computing the gamma and beta weights for the FiLM.
            gamma, beta = film(avg_channel.squeeze()).mean(dim=0).chunk(chunks=2)

            # Expanding tensor back into the correct dimension size.
            gamma = gamma[None, :, None, None].expand_as(z)
            beta = beta[None, :, None, None].expand_as(z)

            # Applying the scale and shift FiLM to the pre-activation output.
            z = (1 + gamma) * z + beta

        return z

    def initialize(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight, 0, 0.01)
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, 0, 0.01)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

        # Initializing the warp layer.
        torch.nn.init.dirac_(self.warp_conv.weight)

    def meta_parameters(self):
        yield from self.conv1.parameters()
        yield from self.conv2.parameters()
        yield from self.conv3.parameters()
        yield from self.warp_conv.parameters()
        yield from self.res_conv.parameters()
        yield from self.film1.parameters()
        yield from self.film2.parameters()
        yield from self.film3.parameters()
        yield from self.warp_film.parameters()

    def base_parameters(self):
        yield from self.conv1.parameters()
        yield from self.conv2.parameters()
        yield from self.conv3.parameters()
        yield from self.res_conv.parameters()


class _PermutationInvariantClassifier(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(_PermutationInvariantClassifier, self).__init__()

        # Creating and initializing the permutation invariant head of the network.
        self.output_layer = torch.nn.Linear(in_features, out_features)  # Placeholder layer.
        self.output_cone = torch.nn.Linear(in_features, 1)  # Classifier weights.

        # Recording the number of input and output features in the classifier.
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        # Computing a forward pass on the linear classifier layer.
        return self.output_layer(x)

    def initialize(self):
        torch.nn.init.normal_(self.output_layer.weight, 0, 0.01)
        torch.nn.init.normal_(self.output_cone.weight, 0, 0.01)
        self.output_layer.bias.data.zero_()
        self.output_cone.bias.data.zero_()

    def reset_classifier(self):
        # Generating the permutation invariant head by copying output cone into the output layer.
        self.output_layer.weight.data = self.output_cone.weight.data.repeat(self.out_features, 1)
        self.output_layer.bias.data = self.output_cone.bias.data.repeat(self.out_features)

    def meta_parameters(self):
        yield from self.output_cone.parameters()

    def base_parameters(self):
        yield from self.output_layer.parameters()


# ============================================================
# Model Variants.
# ============================================================

class AdaResNet(_AdaResNet):

    def __init__(self, **kwargs):
        super(AdaResNet, self).__init__(block_config=[64, 128, 256, 512], **kwargs)


class WideAdaResNet(_AdaResNet):

    def __init__(self, **kwargs):
        super(WideAdaResNet, self).__init__(block_config=[64, 160, 320, 640], **kwargs)
