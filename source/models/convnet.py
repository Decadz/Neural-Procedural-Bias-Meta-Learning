import collections
import torch


class _Conv(torch.nn.Module):

    def __init__(self, input_channels=1, num_filters=32, track_running_stats=False, num_ways=5, **kwargs):
        super(_Conv, self).__init__()

        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("block1", _ConvBlock(input_channels, num_filters, track_running_stats=track_running_stats)),
            ("block2", _ConvBlock(num_filters, num_filters, track_running_stats=track_running_stats)),
            ("block3", _ConvBlock(num_filters, num_filters, track_running_stats=track_running_stats)),
            ("block4", _ConvBlock(num_filters, num_filters, track_running_stats=track_running_stats)),
            ("adaPool", torch.nn.AdaptiveAvgPool2d(1)),
            ("flatten", torch.nn.Flatten())
        ]))

        # Creating and initializing the head of the network.
        self.output_layer = torch.nn.Linear(num_filters, num_ways)

        # Model configuration hyper-parameters.
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_ways = num_ways

        # Initializing the model's parameters.
        self.initialize()

    def forward(self, x):
        x = self.encoder(x)
        return self.output_layer(x)

    def initialize(self):
        # Initializing the encode network.
        for name, module in self.encoder.named_children():
            if isinstance(module, _ConvBlock):
                module.initialize()

        # Initializing the head of the network.
        torch.nn.init.normal_(self.output_layer.weight, 0, 0.01)
        self.output_layer.bias.data.zero_()

    def reset_batch_norm(self):
        # Method for resetting the batch norm statistics.
        for module in self.modules():
            if hasattr(module, 'reset_running_stats'):
                module.reset_running_stats()

    def adapt_parameters(self):
        return self.parameters()

    def warp_parameters(self):
        return self.parameters()


class _WarpConv4(torch.nn.Module):

    def __init__(self, nonlinear=False, input_channels=3, num_filters=32,
                 track_running_stats=False, num_ways=5, **kwargs):

        super(_WarpConv4, self).__init__()

        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("adapt1", _ConvBlock(input_channels, num_filters, track_running_stats=track_running_stats)),
            ("warp1", _WarpBlock(num_filters, num_filters, nonlinearity=nonlinear,
                                 batch_norm=nonlinear, track_running_stats=track_running_stats)),
            ("adapt2", _ConvBlock(num_filters, num_filters, track_running_stats=track_running_stats)),
            ("warp2", _WarpBlock(num_filters, num_filters, nonlinearity=nonlinear,
                                 batch_norm=nonlinear, track_running_stats=track_running_stats)),
            ("adapt3", _ConvBlock(num_filters, num_filters, track_running_stats=track_running_stats)),
            ("warp3", _WarpBlock(num_filters, num_filters, nonlinearity=nonlinear,
                                 batch_norm=nonlinear, track_running_stats=track_running_stats)),
            ("adapt4", _ConvBlock(num_filters, num_filters, track_running_stats=track_running_stats)),
            ("warp4", _WarpBlock(num_filters, num_filters, nonlinearity=nonlinear,
                                 batch_norm=nonlinear, track_running_stats=track_running_stats)),
            ("adaPool", torch.nn.AdaptiveAvgPool2d(1)),
            ("flatten", torch.nn.Flatten())
        ]))

        # Creating and initializing the head of the network.
        self.output_layer = torch.nn.Linear(num_filters, num_ways)

        # Model configuration hyper-parameters.
        self.track_running_stats = track_running_stats
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.nonlinear = nonlinear
        self.num_ways = num_ways

        # Initializing the model's parameters.
        self.initialize()

    def forward(self, x):
        x = self.encoder(x)
        return self.output_layer(x)

    def initialize(self):
        # Initializing the encode network.
        for name, module in self.encoder.named_children():
            if isinstance(module, _ConvBlock) or isinstance(module, _WarpBlock):
                module.initialize()

        # Initializing the head of the network.
        torch.nn.init.normal_(self.output_layer.weight, 0, 0.01)
        self.output_layer.bias.data.zero_()

    def reset_batch_norm(self):
        # Method for resetting the batch norm statistics.
        for module in self.modules():
            if hasattr(module, 'reset_running_stats'):
                module.reset_running_stats()

    def adapt_parameters(self):
        for param in self.encoder.adapt1.parameters():
            yield param
        for param in self.encoder.adapt2.parameters():
            yield param
        for param in self.encoder.adapt3.parameters():
            yield param
        for param in self.encoder.adapt4.parameters():
            yield param
        for param in self.output_layer.parameters():
            yield param

    def warp_parameters(self):
        for param in self.encoder.warp1.parameters():
            yield param
        for param in self.encoder.warp2.parameters():
            yield param
        for param in self.encoder.warp3.parameters():
            yield param
        for param in self.encoder.warp4.parameters():
            yield param


# ============================================================
# Network block definitions.
# ============================================================


class _ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, track_running_stats=False):
        super(_ConvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=track_running_stats)
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.pool(x)

    def initialize(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight, 0, 0.01)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _WarpBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, nonlinearity=True, batch_norm=True,
                 stacked_conv=False, residual_connection=False, track_running_stats=False):

        super(_WarpBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

        if batch_norm:
            self.bn_in = torch.nn.BatchNorm2d(in_channels, track_running_stats=track_running_stats)

        self.activation1 = torch.nn.ReLU(inplace=True) if nonlinearity else torch.nn.Identity()

        if stacked_conv:
            self.conv2 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.activation2 = torch.nn.ReLU(inplace=True) if nonlinearity else torch.nn.Identity()

        if batch_norm and residual_connection:
            self.bn_out = torch.nn.BatchNorm2d(out_channels, track_running_stats=track_running_stats)

        self.nonlinearity = nonlinearity
        self.batch_norm = batch_norm
        self.stacked_conv = stacked_conv
        self.residual_connection = residual_connection

    def forward(self, x):
        h = x

        h = self.conv1(h)

        if self.batch_norm:
            h = self.bn_in(h)

        h = self.activation1(h)

        if self.stacked_conv:
            h = self.conv2(h)
            h = self.activation2(h)

        if self.residual_connection:
            h = x + h

        if self.batch_norm and self.residual_connection:
            h = self.bn_out(h)

        return h

    def initialize(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.dirac_(module.weight)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


# ============================================================
# Model Variants.
# ============================================================


class Conv(_Conv):

    def __init__(self, **kwargs):
        super(Conv, self).__init__(num_filters=32, **kwargs)


class WideConv(_Conv):

    def __init__(self, **kwargs):
        super(WideConv, self).__init__(num_filters=64, **kwargs)


class LinearWarpConv(_WarpConv4):

    def __init__(self, **kwargs):
        super(LinearWarpConv, self).__init__(num_filters=32, nonlinear=False, **kwargs)


class LinearWarpWideConv(_WarpConv4):

    def __init__(self, **kwargs):
        super(LinearWarpWideConv, self).__init__(num_filters=64, nonlinear=False, **kwargs)


class WarpConv(_WarpConv4):

    def __init__(self, **kwargs):
        super(WarpConv, self).__init__(num_filters=32, nonlinear=True, **kwargs)


class WarpWideConv(_WarpConv4):

    def __init__(self, **kwargs):
        super(WarpWideConv, self).__init__(num_filters=64, nonlinear=True, **kwargs)
