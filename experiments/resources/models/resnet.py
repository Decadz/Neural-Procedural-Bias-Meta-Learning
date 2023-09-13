import collections
import torch


class ResNet12(torch.nn.Module):

    def __init__(self, input_channels=3, num_ways=5, **kwargs):

        """
        Implementation of ResNets from the paper "Deep Residual Learning
        for Image Recognition" by Kaiming He, Xiangyu Zhang, Shaoqing
        Ren, and Jian Sun.
        """

        super(ResNet12, self).__init__()
        channels = [64, 128, 256, 512]  # wrn = [64, 160, 320, 640]

        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("block1", Block(input_channels, channels[0])),
            ("block2", Block(channels[0], channels[1])),
            ("block3", Block(channels[1], channels[2])),
            ("block4", Block(channels[2], channels[3])),
            ("adaPool", torch.nn.AdaptiveAvgPool2d(1)),
            ("flatten", torch.nn.Flatten())
        ]))

        self.output_layer = torch.nn.Linear(512, num_ways)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1.)
                torch.nn.init.constant_(m.bias, 0.)

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


class Block(torch.nn.Module):

    def __init__(self, in_planes, planes):
        super(Block, self).__init__()
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
