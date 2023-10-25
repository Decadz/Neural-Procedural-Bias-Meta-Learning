import collections
import torch


class Conv4(torch.nn.Module):

    def __init__(self, input_channels=1, num_filters=32, num_ways=5, **kwargs):
        super(Conv4, self).__init__()

        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("block1", _ConvBlock(input_channels, num_filters)),
            ("block2", _ConvBlock(num_filters, num_filters)),
            ("block3", _ConvBlock(num_filters, num_filters)),
            ("block4", _ConvBlock(num_filters, num_filters)),
            ("flatten", torch.nn.Flatten())
        ]))

        linear_in = 800 if input_channels == 3 else 32
        self.output_layer = torch.nn.Linear(linear_in, num_ways)

        # Model configuration hyper-parameters.
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_ways = num_ways

        # Initializing the model's parameters.
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight, 0, 0.01)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, 0, 0.01)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        return self.output_layer(x)
    
    def adapt_parameters(self):
        for param in self.encoder.block1.parameters():
            yield param
        for param in self.encoder.block3.parameters():
            yield param
        for param in self.output_layer.parameters():
            yield param
        
    def warp_parameters(self):
        for param in self.encoder.block2.parameters():
            yield param
        for param in self.encoder.block4.parameters():
            yield param

    def init_adaptation(self):
        # Function for resetting the batch norm statistics.
        for m in self.modules():
            if hasattr(m, 'reset_running_stats'):
                m.reset_running_stats()


class LinearWarpConv4(torch.nn.Module):

    def __init__(self, input_channels=3, num_filters=32, num_ways=5, **kwargs):
        super(LinearWarpConv4, self).__init__()

        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("adapt1", _ConvBlock(input_channels, num_filters)),
            ("warp1", _WarpBlock(num_filters, num_filters, nonlinearity=False, batch_norm=False,
                                 stacked_conv=False, residual_connection=False)),
            ("adapt2", _ConvBlock(num_filters, num_filters)),
            ("warp2", _WarpBlock(num_filters, num_filters, nonlinearity=False, batch_norm=False,
                                 stacked_conv=False, residual_connection=False)),
            ("adapt3", _ConvBlock(num_filters, num_filters)),
            ("warp3", _WarpBlock(num_filters, num_filters, nonlinearity=False, batch_norm=False,
                                 stacked_conv=False, residual_connection=False)),
            ("adapt4", _ConvBlock(num_filters, num_filters)),
            ("warp4", _WarpBlock(num_filters, num_filters, nonlinearity=False, batch_norm=False,
                                 stacked_conv=False, residual_connection=False)),
            ("flatten", torch.nn.Flatten())
        ]))

        linear_in = 800 if input_channels == 3 else 32
        self.output_layer = torch.nn.Linear(linear_in, num_ways)

        # Model configuration hyper-parameters.
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_ways = num_ways

        # Initializing the model's parameters.
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight, 0, 0.01)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, 0, 0.01)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        return self.output_layer(x)

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


class WarpConv4(torch.nn.Module):

    def __init__(self, input_channels=3, num_filters=32, num_ways=5, **kwargs):
        super(WarpConv4, self).__init__()

        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("adapt1", _ConvBlock(input_channels, num_filters)),
            ("warp1", _WarpBlock(num_filters, num_filters, stacked_conv=False, residual_connection=False)),
            ("adapt2", _ConvBlock(num_filters, num_filters)),
            ("warp2", _WarpBlock(num_filters, num_filters, stacked_conv=False, residual_connection=False)),
            ("adapt3", _ConvBlock(num_filters, num_filters)),
            ("warp3", _WarpBlock(num_filters, num_filters, stacked_conv=False, residual_connection=False)),
            ("adapt4", _ConvBlock(num_filters, num_filters)),
            ("warp4", _WarpBlock(num_filters, num_filters, stacked_conv=False, residual_connection=False)),
            ("flatten", torch.nn.Flatten())
        ]))

        linear_in = 800 if input_channels == 3 else 32
        self.output_layer = torch.nn.Linear(linear_in, num_ways)

        # Model configuration hyper-parameters.
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_ways = num_ways

        # Initializing the model's parameters.
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.normal_(module.weight, 0, 0.01)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, 0, 0.01)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        return self.output_layer(x)
    
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


class _ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_ConvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.relu = torch.nn.LeakyReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.pool(x)


class _WarpBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, nonlinearity=True, batch_norm=True,
                  stacked_conv=True, residual_connection=True):
        
        super(_WarpBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

        if batch_norm:
            self.bn_in = torch.nn.BatchNorm2d(in_channels, track_running_stats=True)

        self.activation1 = torch.nn.LeakyReLU(inplace=True) if nonlinearity else torch.nn.Identity()

        if stacked_conv:
            self.conv2 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.activation2 = torch.nn.LeakyReLU(inplace=True) if nonlinearity else torch.nn.Identity()

        if batch_norm and residual_connection:
            self.bn_out = torch.nn.BatchNorm2d(out_channels, track_running_stats=True)

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
