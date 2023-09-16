import collections
import torch


class Conv4(torch.nn.Module):

    def __init__(self, input_channels=1, num_filters=64, num_ways=5, **kwargs):
        super(Conv4, self).__init__()

        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("block1", _ConvBlock(input_channels, num_filters)),
            ("block2", _ConvBlock(num_filters, num_filters)),
            ("block3", _ConvBlock(num_filters, num_filters)),
            ("block4", _ConvBlock(num_filters, num_filters)),
        ]))

        linear_in = 1600 if input_channels == 3 else 64
        self.output_layer = torch.nn.Linear(linear_in, num_ways)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
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


class WarpConv4(torch.nn.Module):

    def __init__(self, input_channels=3, num_filters=128, num_ways=5, **kwargs):
        super(WarpConv4, self).__init__()

        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("adapt1", _ConvBlock(input_channels, num_filters)),
            ("adapt2", _ConvBlock(num_filters, num_filters)),
            ("warp2", _WarpBlock(num_filters, num_filters)),
            ("adapt3", _ConvBlock(num_filters, num_filters)),
            ("warp3", _WarpBlock(num_filters, num_filters)),
            ("adapt4", _ConvBlock(num_filters, num_filters)),
            ("warp4", _WarpBlock(num_filters, num_filters)),
        ]))

        linear_in = 3200 if input_channels == 3 else 128
        self.output_layer = torch.nn.Linear(linear_in, num_ways)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
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
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
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
        
        if batch_norm:
            self.bn_in = torch.nn.BatchNorm2d(in_channels)

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.activation1 = torch.nn.ReLU(inplace=True) if nonlinearity else torch.nn.Identity()

        if stacked_conv:
            self.conv2 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.activation2 = torch.nn.ReLU(inplace=True) if nonlinearity else torch.nn.Identity()

        if batch_norm and residual_connection:
            self.bn_out = torch.nn.BatchNorm2d(out_channels)

        self.nonlinearity = nonlinearity
        self.batch_norm = batch_norm
        self.stacked_conv = stacked_conv
        self.residual_connection = residual_connection
        

    def forward(self, x):
        h = x

        if self.batch_norm:
            h = self.bn_in(h)

        h = self.conv1(h)
        h = self.activation1(h)

        if self.stacked_conv:
            h = self.conv2(h)
            h = self.activation2(h)

        if self.residual_connection:
            h = x + h

        if self.batch_norm and self.residual_connection:
            h = self.bn_out(h)

        return h
