import torch


class Conv4a(torch.nn.Module):

    def __init__(self, input_channels=1, num_filters=64, num_ways=5, **kwargs):
        super(Conv4a, self).__init__()

        self.encoder = torch.nn.Sequential(
            _ConvBlock(input_channels, num_filters),
            _ConvBlock(num_filters, num_filters),
            _ConvBlock(num_filters, num_filters),
            _ConvBlock(num_filters, num_filters),
        )

        self.output_layer = torch.nn.Linear(num_filters, num_ways)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        return self.output_layer(x)


class Conv4b(torch.nn.Module):

    def __init__(self, input_channels=3, num_filters=64, num_ways=5, **kwargs):
        super(Conv4b, self).__init__()

        self.encoder = torch.nn.Sequential(
            _ConvBlock(input_channels, num_filters),
            _ConvBlock(num_filters, num_filters),
            _ConvBlock(num_filters, num_filters),
            _ConvBlock(num_filters, num_filters),
        )

        self.output_layer = torch.nn.Linear(1600, num_ways)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        return self.output_layer(x)


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
