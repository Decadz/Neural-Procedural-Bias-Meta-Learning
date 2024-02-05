import collections
import torch


class _AdaConv(torch.nn.Module):

    def __init__(self, input_channels=1, num_filters=32, num_ways=5, **kwargs):
        super(_AdaConv, self).__init__()

        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("adapt1", _ConvBlock(input_channels, num_filters)),
            ("adapt2", _ConvBlock(num_filters, num_filters)),
            ("adapt3", _FiLMConvBlock(num_filters, num_filters)),
            ("warp3", _FiLMWarpBlock(num_filters, num_filters)),
            ("adapt4", _FiLMConvBlock(num_filters, num_filters)),
            ("warp4", _FiLMWarpBlock(num_filters, num_filters)),
            ("adaPool", torch.nn.AdaptiveAvgPool2d(1)),
            ("flatten", torch.nn.Flatten())
        ]))

        # Creating the permutation invariant head of the network.
        self.classifier = _PermutationInvariantClassifier(num_filters, num_ways)

        # Model configurations and hyper-parameters.
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_ways = num_ways

        # Initializing the model's parameters.
        self.initialize()

    def forward(self, x):

        # Generating the image embeddings using the encoder.
        z = self.encoder(x)

        # Generating the model predictions.
        return self.classifier(z), z

    def initialize(self):
        # Initializing the networks parameters.
        self.classifier.initialize()
        for name, module in self.encoder.named_children():
            if isinstance(module, (_ConvBlock, _FiLMConvBlock, _FiLMWarpBlock)):
                module.initialize()

    def reset_classifier(self):
        # Resetting the output layer using the output cone.
        self.classifier.reset_classifier()

    def meta_parameters(self):
        for module in self.encoder.children():
            if hasattr(module, "meta_parameters"):
                yield from module.meta_parameters()
        yield from self.classifier.meta_parameters()

    def base_parameters(self):
        yield from self.encoder.adapt3.base_parameters()
        yield from self.encoder.adapt4.base_parameters()
        yield from self.classifier.base_parameters()

    def pretraining_parameters(self):
        for module in self.encoder.children():
            if hasattr(module, "base_parameters"):
                yield from module.base_parameters()
        yield from self.classifier.base_parameters()

# ============================================================
# Network block definitions.
# ============================================================


class _ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock, self).__init__()

        # The underlying convolutional layer.
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

        # Batch normalization layer to reduce internal covariance shift.
        self.bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)

        # The non-linear activation function.
        self.relu = torch.nn.ReLU(inplace=True)

        # The pooling layer used for down-sampling the output volume.
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):

        # Computing a forward pass on the convolutional layer.
        z = self.bn(self.conv(x))

        # Applying the non-linear activation function and down-sample representation.
        return self.pool(self.relu(z))

    def initialize(self):
        torch.nn.init.normal_(self.conv.weight, 0, 0.01)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def meta_parameters(self):
        yield from self.conv.parameters()

    def base_parameters(self):
        yield from self.conv.parameters()


class _FiLMConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_FiLMConvBlock, self).__init__()

        # The underlying convolutional layer.
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

        # Batch normalization layer to reduce internal covariance shift.
        self.bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)

        # The feature wise linear modulation (FiLM) layer for making the layer adaptive.
        self.film = torch.nn.Linear(in_channels, out_channels * 2)

        # The non-linear activation function.
        self.relu = torch.nn.ReLU(inplace=True)

        # The pooling layer used for down-sampling the output volume.
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):

        # Computing a forward pass on the convolutional layer.
        z = self.bn(self.conv(x))

        # Computing the average value for each channel in the volume.
        avg_channel = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))

        # Computing the gamma and beta weights and mean reducing in the batch dimension.
        gamma, beta = self.film(avg_channel.squeeze()).mean(dim=0).chunk(2)

        # Expanding tensor back into the correct dimension size.
        gamma = gamma[None, :, None, None].expand_as(z)
        beta = beta[None, :, None, None].expand_as(z)

        return self.pool(self.relu((1 + gamma) * z + beta))

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


class _FiLMWarpBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_FiLMWarpBlock, self).__init__()

        # The underlying convolutional layer (as implemented in PyTorch).
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

        # Batch normalization layer to reduce internal covariance shift.
        self.bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)

        # The feature wise linear modulation (FiLM) layer for making the layer adaptive.
        self.film = torch.nn.Linear(in_channels, out_channels * 2)

    def forward(self, x):

        # Computing a forward pass on the convolutional layer.
        z = self.bn(self.conv(x))

        # Computing the average value for each channel in the volume.
        avg_channel = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))

        # Computing the gamma and beta weights and mean reducing in the batch dimension.
        gamma, beta = self.film(avg_channel.squeeze()).mean(dim=0).chunk(2)

        # Expanding tensor back into the correct dimension size.
        gamma = gamma[None, :, None, None].expand_as(z)
        beta = beta[None, :, None, None].expand_as(z)

        return (1 + gamma) * z + beta

    def initialize(self):
        torch.nn.init.dirac_(self.conv.weight)
        torch.nn.init.normal_(self.film.weight, 0, 0.01)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def meta_parameters(self):
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


class AdaConv32(_AdaConv):

    def __init__(self, **kwargs):
        super(AdaConv32, self).__init__(num_filters=32, **kwargs)


class AdaConv48(_AdaConv):

    def __init__(self, **kwargs):
        super(AdaConv48, self).__init__(num_filters=48, **kwargs)


class AdaConv64(_AdaConv):

    def __init__(self, **kwargs):
        super(AdaConv64, self).__init__(num_filters=64, **kwargs)


class AdaConv128(_AdaConv):

    def __init__(self, **kwargs):
        super(AdaConv128, self).__init__(num_filters=128, **kwargs)
