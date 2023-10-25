import collections
import torch


class _ResNet(torch.nn.Module):

    def __init__(self, block_config, input_channels=3, num_ways=5, **kwargs):

        """
        Implementation of ResNets from the paper "Deep Residual Learning
        for Image Recognition" by Kaiming He, Xiangyu Zhang, Shaoqing
        Ren, and Jian Sun.
        """

        super(_ResNet, self).__init__()

        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("block1", _Block(input_channels, block_config[0])),
            ("block2", _Block(block_config[0], block_config[1])),
            ("block3", _Block(block_config[1], block_config[2])),
            ("block4", _Block(block_config[2], block_config[3])),
            ("adaPool", torch.nn.AdaptiveAvgPool2d(1)),
            ("flatten", torch.nn.Flatten())
        ]))

        self.output_layer = torch.nn.Linear(block_config[-1], num_ways)

        # Model configuration hyper-parameters.
        self.input_channels = input_channels
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


class _WarpResNet(torch.nn.Module):

    def __init__(self, block_config, input_channels=3, num_ways=5, **kwargs):
        super(_WarpResNet, self).__init__()

        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("block1", _Block(input_channels, block_config[0])),
            ("warp1", _Block(block_config[0], block_config[0])),
            ("block2", _Block(block_config[0], block_config[1])),
            ("warp2", _Block(block_config[1], block_config[1])),
            ("block3", _Block(block_config[1], block_config[2])),
            ("warp3", _Block(block_config[2], block_config[2])),
            ("block4", _Block(block_config[2], block_config[3])),
            ("warp4", _Block(block_config[3], block_config[3])),
            ("adaPool", torch.nn.AdaptiveAvgPool2d(1)),
            ("flatten", torch.nn.Flatten())
        ]))

        self.output_layer = torch.nn.Linear(block_config[-1], num_ways)

        # Model configuration hyper-parameters.
        self.input_channels = input_channels
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


class _Block(torch.nn.Module):

    def __init__(self, in_planes, planes):
        super(_Block, self).__init__()
        self.in_planes = in_planes
        self.planes = planes

        self.conv1 = torch.nn.Conv2d(in_planes, planes, 3, 1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, 3, 1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, planes, 3, 1, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes)

        self.res_conv = torch.nn.Conv2d(in_planes, planes, 1, 1, padding=0, bias=False)
        self.bn = torch.nn.BatchNorm2d(planes)

        self.relu = torch.nn.LeakyReLU(0.1, inplace=True)
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        x = self.res_conv(x)
        x = self.bn(x)
        out = self.pool(self.relu(out + x))
        return out


class ResNet(_ResNet):

    def __init__(self, **kwargs):
        super(ResNet, self).__init__(block_config=[64, 128, 256, 512], **kwargs)


class WideResNet(_ResNet):

    def __init__(self, **kwargs):
        super(WideResNet, self).__init__(block_config=[64, 160, 320, 640], **kwargs)


class WarpResNet(_WarpResNet):

    def __init__(self, **kwargs):
        super(WarpResNet, self).__init__(block_config=[64, 128, 256, 512], **kwargs)


class WarpWideResNet(_WarpResNet):

    def __init__(self, **kwargs):
        super(WarpWideResNet, self).__init__(block_config=[64, 160, 320, 640], **kwargs)
