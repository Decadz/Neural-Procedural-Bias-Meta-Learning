import collections
import torch


class _ResNet(torch.nn.Module):

    def __init__(self, block_config, input_channels=3, num_ways=5, **kwargs):
        super(_ResNet, self).__init__()

        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("adapt1", _ConvBlock(input_channels, block_config[0])),
            ("adapt2", _ConvBlock(block_config[0], block_config[1])),
            ("adapt3", _ConvBlock(block_config[1], block_config[2])),
            ("adapt4", _ConvBlock(block_config[2], block_config[3])),
            ("adaPool", torch.nn.AdaptiveAvgPool2d(1)),
            ("flatten", torch.nn.Flatten())
        ]))

        # Creating and initializing the head of the network.
        self.classifier = _Classifier(block_config[-1], num_ways)

        # Model configuration hyper-parameters.
        self.input_channels = input_channels
        self.block_config = block_config
        self.num_ways = num_ways

        # Initializing the model's parameters.
        self.initialize()

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)

    def initialize(self):
        # Initializing the networks parameters.
        self.classifier.initialize()
        for name, module in self.encoder.named_children():
            if isinstance(module, _ConvBlock):
                module.initialize()

    def meta_parameters(self):
        for module in self.encoder.children():
            if hasattr(module, "meta_parameters"):
                yield from module.meta_parameters()
        yield from self.classifier.meta_parameters()

    def base_parameters(self):
        for module in self.encoder.children():
            if hasattr(module, "base_parameters"):
                yield from module.base_parameters()
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

        # The convolutional feature extractor block, containing three conv -> bn.
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
        # Initializing the sequential block.
        for name, module in self.block.named_children():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight, 0, 0.01)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

        torch.nn.init.normal_(self.res_conv.weight, 0, 0.01)
        self.res_bn.weight.data.fill_(1)
        self.res_bn.bias.data.zero_()

    def meta_parameters(self):
        for name, module in self.block.named_children():
            if isinstance(module, torch.nn.Conv2d):
                yield from module.parameters()
        yield from self.res_conv.parameters()

    def base_parameters(self):
        for name, module in self.block.named_children():
            if isinstance(module, torch.nn.Conv2d):
                yield from module.parameters()
        yield from self.res_conv.parameters()


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


class ResNet(_ResNet):

    def __init__(self, **kwargs):
        super(ResNet, self).__init__(block_config=[64, 128, 256, 512], **kwargs)


class WideResNet(_ResNet):

    def __init__(self, **kwargs):
        super(WideResNet, self).__init__(block_config=[64, 160, 320, 640], **kwargs)
