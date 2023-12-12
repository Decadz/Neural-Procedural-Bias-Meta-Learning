import collections
import torch


class _Conv(torch.nn.Module):

    def __init__(self, input_channels=1, num_filters=32, num_ways=5, **kwargs):
        super(_Conv, self).__init__()

        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("adapt1", _ConvBlock(input_channels, num_filters)),
            ("warp1", _WarpBlock(num_filters, num_filters)),
            ("adapt2", _ConvBlock(num_filters, num_filters)),
            ("warp2", _WarpBlock(num_filters, num_filters)),
            ("adapt3", _ConvBlock(num_filters, num_filters)),
            ("warp3", _WarpBlock(num_filters, num_filters)),
            ("adapt4", _ConvBlock(num_filters, num_filters)),
            ("warp4", _WarpBlock(num_filters, num_filters)),
            ("adaPool", torch.nn.AdaptiveAvgPool2d(1)),
            ("flatten", torch.nn.Flatten())
        ]))

        # Creating and initializing the head of the network.
        self.classifier = _Classifier(num_filters, num_ways)

        # Model configuration hyper-parameters.
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_ways = num_ways

        # Initializing the model's parameters.
        self.initialize()

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)

    def initialize(self):
        # Initializing the networks parameters.
        self.classifier.initialize()
        for name, module in self.encoder.named_children():
            if isinstance(module, (_ConvBlock, _WarpBlock)):
                module.initialize()

    def reset_batch_norm(self):
        # Method for resetting the batch norm statistics.
        for module in self.modules():
            if hasattr(module, 'reset_running_stats'):
                module.reset_running_stats()

    def meta_parameters(self):
        for module in self.encoder.children():
            if hasattr(module, "meta_parameters"):
                yield from module.meta_parameters()
        yield from self.classifier.meta_parameters()

    def warp_parameters(self):
        for module in self.encoder.children():
            if hasattr(module, "warp_parameters"):
                yield from module.warp_parameters()

    def base_parameters(self):
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
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True)
        self.bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.pool(x)

    def initialize(self):
        torch.nn.init.normal_(self.conv.weight, 0, 0.01)
        self.conv.bias.data.zero_()
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def meta_parameters(self):
        yield from self.conv.parameters()

    def base_parameters(self):
        yield from self.conv.parameters()


class _WarpBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_WarpBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True)
        self.bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)

    def forward(self, x):
        return self.bn(self.conv(x))

    def initialize(self):
        torch.nn.init.dirac_(self.conv.weight)
        self.conv.bias.data.zero_()
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def warp_parameters(self):
        yield from self.conv.parameters()


class _Classifier(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(_Classifier, self).__init__()

        # Creating a permutation variant classifier head (i.e. not unicorn head).
        self.output_layer = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        # Computing the forward pass on the linear classifier layer.
        return self.output_layer(x)

    def initialize(self):
        torch.nn.init.normal_(self.output_layer.weight, 0, 0.01)
        self.output_layer.bias.data.zero_()

    def meta_parameters(self):
        yield from self.output_layer.parameters()

    def base_parameters(self):
        yield from self.output_layer.parameters()


# ============================================================
# Model Variants.
# ============================================================

class Conv32(_Conv):

    def __init__(self, **kwargs):
        super(Conv32, self).__init__(num_filters=32, **kwargs)


class Conv48(_Conv):

    def __init__(self, **kwargs):
        super(Conv48, self).__init__(num_filters=48, **kwargs)


class Conv64(_Conv):

    def __init__(self, **kwargs):
        super(Conv64, self).__init__(num_filters=64, **kwargs)


class Conv128(_Conv):

    def __init__(self, **kwargs):
        super(Conv128, self).__init__(num_filters=128, **kwargs)
