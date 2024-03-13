import collections
import torch


class _AdaResNet(torch.nn.Module):

    def __init__(self, block_config, input_channels=3, num_ways=5, **kwargs):
        super(_AdaResNet, self).__init__()

        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("block1", _ConvBlock(input_channels, block_config[0])),
            ("block2", _ConvBlock(block_config[0], block_config[1])),
            ("block3", _ConvBlock(block_config[1], block_config[2])),
            ("adapt", _FiLMConvBlock(block_config[2], block_config[3])),
            ("warp", _FiLMWarpBlock(block_config[3], block_config[3])),
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
            if isinstance(module, (_FiLMConv, _FiLMConvBlock, _FiLMWarpBlock)):
                module.task_adaptive = task_adaptive

        # Generating the image embeddings using the encoder.
        z = self.encoder(x)

        # Generating the model predictions.
        return self.classifier(z)

    def initialize(self):

        # Initializing the networks parameters.
        self.classifier.initialize()
        for name, module in self.encoder.named_children():
            if isinstance(module, (_FiLMConvBlock, _FiLMWarpBlock)):
                module.initialize()

    def reset_classifier(self):
        # Resetting the output layer using the output cone.
        self.classifier.reset_classifier()

    def meta_parameters(self):
        for name, module in self.encoder.named_children():
            if hasattr(module, "meta_parameters"):
                if isinstance(module, (_FiLMConvBlock, _FiLMWarpBlock)):
                    yield from module.meta_parameters()
        yield from self.classifier.meta_parameters()

    def base_parameters(self):
        for name, module in self.encoder.named_children():
            if hasattr(module, "base_parameters"):
                if isinstance(module, _FiLMConvBlock):
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

        # The convolutional feature extractor block, containing three filmed conv -> bn.
        self.block = torch.nn.Sequential(collections.OrderedDict([
            ("adapt1", torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)),
            ("bn1", torch.nn.BatchNorm2d(out_channels, track_running_stats=False)),
            ("relu1", torch.nn.ReLU(inplace=True)),
            ("adapt2", torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)),
            ("bn2", torch.nn.BatchNorm2d(out_channels, track_running_stats=False)),
            ("relu2", torch.nn.ReLU(inplace=True)),
            ("adapt3", torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)),
            ("bn3", torch.nn.BatchNorm2d(out_channels, track_running_stats=False)),
        ]))

        # Residual (skip) connections. Cant been in sequential block since it runs in parallel.
        self.res_conv = torch.nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=False)
        self.res_bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)

        # Block non-linearity and down-sampling layer.
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(2)

        # Block configurations and hyper-parameters.
        self.in_planes = in_channels
        self.planes = out_channels

    def forward(self, x):
        out = self.block(x)
        res = self.res_bn(self.res_conv(x))
        return self.pool(self.relu(out + res))

    def initialize(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight, 0, 0.01)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def base_parameters(self):
        yield from self.block.parameters()
        yield from self.res_conv.parameters()


class _FiLMConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_FiLMConvBlock, self).__init__()

        # The convolutional feature extractor block, containing three filmed conv -> bn.
        self.block = torch.nn.Sequential(collections.OrderedDict([
            ("adapt1", _FiLMConv(in_channels, out_channels)),
            ("relu1", torch.nn.ReLU(inplace=True)),
            ("adapt2", _FiLMConv(out_channels, out_channels)),
            ("relu2", torch.nn.ReLU(inplace=True)),
            ("adapt3", _FiLMConv(out_channels, out_channels)),
        ]))

        # Residual (skip) connections. Cant been in sequential block since it runs in parallel.
        self.res_conv = torch.nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, bias=False)
        self.res_bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.res_film = torch.nn.Linear(out_channels, out_channels * 2)

        # Block non-linearity and down-sampling layer.
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(2)

        # Block configurations and hyper-parameters.
        self.in_planes = in_channels
        self.planes = out_channels

        # Field for controlling the current state of the layer.
        self.task_adaptive = False

    def forward(self, x):
        out = self.block(x)
        res = self.res_bn(self.res_conv(x))

        if self.task_adaptive:  # If task adaptive apply Feature Wise Linear Modulation (FiLM).

            # Computing the local and global embeddings.
            avg_channel = torch.nn.functional.adaptive_avg_pool2d(res, (1, 1))

            # Computing the gamma and beta weights for the FiLM.
            gamma, beta = self.res_film(avg_channel.squeeze()).chunk(chunks=2, dim=1)

            # Expanding in the spatial (width and height) dimension.
            gamma = gamma[:, :, None, None].expand_as(res)
            beta = beta[:, :, None, None].expand_as(res)

            # Applying the scale and shift FiLM to the pre-activation output.
            res = (1 + gamma) * res + beta

        return self.pool(self.relu(out + res))

    def initialize(self):
        # Initializing the sequential block.
        for name, module in self.block.named_children():
            if isinstance(module, _FiLMConv):
                module.initialize()

        torch.nn.init.normal_(self.res_conv.weight, 0, 0.01)
        torch.nn.init.normal_(self.res_film.weight, 0, 0.01)
        self.res_bn.weight.data.fill_(1)
        self.res_bn.bias.data.zero_()

    def meta_parameters(self):
        for module in self.block.children():
            if hasattr(module, "meta_parameters"):
                yield from module.meta_parameters()

        yield from self.res_conv.parameters()
        yield from self.res_film.parameters()

    def base_parameters(self):
        for module in self.block.children():
            if hasattr(module, "base_parameters"):
                yield from module.base_parameters()

        yield from self.res_conv.parameters()
        yield from self.res_film.parameters()


class _FiLMWarpBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_FiLMWarpBlock, self).__init__()

        # The underlying convolutional layer (as implemented in PyTorch).
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

        # Batch normalization layer to reduce internal covariance shift.
        self.bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)

        # The feature wise linear modulation (FiLM) layer for making the layer adaptive.
        self.film = torch.nn.Linear(in_channels, out_channels * 2)

        # Field for controlling the current state of the layer.
        self.task_adaptive = False

    def forward(self, x):

        # Computing a forward pass on the convolutional layer.
        z = self.bn(self.conv(x))

        if self.task_adaptive:  # If task adaptive apply Feature Wise Linear Modulation (FiLM).

            # Computing the local and global embeddings.
            avg_channel = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))

            # Computing the gamma and beta weights for the FiLM.
            gamma, beta = self.film(avg_channel.squeeze()).chunk(chunks=2, dim=1)

            # Expanding in the spatial (width and height) dimension.
            gamma = gamma[:, :, None, None].expand_as(z)
            beta = beta[:, :, None, None].expand_as(z)

            # Applying the scale and shift FiLM to the pre-activation output.
            z = (1 + gamma) * z + beta

        return z

    def initialize(self):
        torch.nn.init.dirac_(self.conv.weight)
        torch.nn.init.normal_(self.film.weight, 0, 0.01)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def meta_parameters(self):
        yield from self.conv.parameters()
        yield from self.film.parameters()


class _FiLMConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_FiLMConv, self).__init__()

        # The underlying convolutional layer (as implemented in PyTorch).
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

        # Batch normalization layer to reduce internal covariance shift.
        self.bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)

        # The feature wise linear modulation (FiLM) layer for making the layer adaptive.
        self.film = torch.nn.Linear(in_channels, out_channels * 2)

        # Field for controlling the current state of the layer.
        self.task_adaptive = False

    def forward(self, x):

        # Computing a forward pass on the convolutional layer.
        z = self.bn(self.conv(x))

        if self.task_adaptive:  # If task adaptive apply Feature Wise Linear Modulation (FiLM).

            # Computing the local and global embeddings.
            avg_channel = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))

            # Computing the gamma and beta weights for the FiLM.
            gamma, beta = self.film(avg_channel.squeeze()).chunk(chunks=2, dim=1)

            # Expanding in the spatial (width and height) dimension.
            gamma = gamma[:, :, None, None].expand_as(z)
            beta = beta[:, :, None, None].expand_as(z)

            # Applying the scale and shift FiLM to the pre-activation output.
            z = (1 + gamma) * z + beta

        return z

    def initialize(self):
        torch.nn.init.normal_(self.conv.weight, 0, 0.01)
        torch.nn.init.normal_(self.film.weight, 0, 0.01)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def meta_parameters(self):
        yield from self.conv.parameters()
        yield from self.film.parameters()

    def base_parameters(self):
        yield from self.conv.parameters()
        yield from self.film.parameters()


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
