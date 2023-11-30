import collections
import torch


class _Conv(torch.nn.Module):

    def __init__(self, input_channels=1, num_filters=32, num_ways=5, **kwargs):
        super(_Conv, self).__init__()

        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("block1", _FiLMConvBlock(input_channels, num_filters)),
            ("block2", _FiLMConvBlock(num_filters, num_filters)),
            ("block3", _FiLMConvBlock(num_filters, num_filters)),
            ("block4", _FiLMConvBlock(num_filters, num_filters)),
            ("adaPool", torch.nn.AdaptiveAvgPool2d(1)),
            ("flatten", torch.nn.Flatten())
        ]))

        # Creating and initializing the permutation invariant head of the network.
        self.output_layer = torch.nn.Linear(num_filters, num_ways)  # Placeholder layer.
        #self.output_cone = torch.nn.Linear(num_filters, 1)   # Classifier weights.

        # Model configuration hyper-parameters.
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_ways = num_ways

        # Initializing the model's parameters.
        self.initialize()

    def forward(self, x, adapt=False):

        # Setting the settings for all the FiLM layers.
        for name, module in self.encoder.named_children():
            if isinstance(module, _FiLMConvBlock):
                module.adapt = adapt

        # Generating the image embeddings using the encoder.
        x = self.encoder(x)

        # Generating the permutation invariant head by copying output cone into the output layer.
        #self.output_layer.weight.data = self.output_cone.weight.data.repeat(self.num_ways, 1)
        #self.output_layer.bias.data = self.output_cone.bias.data.repeat(self.num_ways)

        # Generating the model predictions.
        return self.output_layer(x)

    def initialize(self):
        # Initializing the encoder network.
        for name, module in self.encoder.named_children():
            if isinstance(module, _FiLMConvBlock):
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
        for param in self.encoder.block1.adapt_parameters():
            yield param
        for param in self.encoder.block2.adapt_parameters():
            yield param
        for param in self.encoder.block3.adapt_parameters():
            yield param
        for param in self.encoder.block4.adapt_parameters():
            yield param
        for param in self.output_layer.parameters():
            yield param

    def film_parameters(self):
        for param in self.encoder.block1.film_parameters():
            yield param
        for param in self.encoder.block2.film_parameters():
            yield param
        for param in self.encoder.block3.film_parameters():
            yield param
        for param in self.encoder.block4.film_parameters():
            yield param

    def warp_parameters(self):
        pass

# ============================================================
# Network block definitions.
# ============================================================


class _FiLMConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_FiLMConvBlock, self).__init__()

        # The underlying convolutional layer (as implemented in PyTorch).
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

        # Batch normalization layer to reduce internal covariance shift.
        self.bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)

        # The feature wise linear modulation (FiLM) layer for making the layer adaptive.
        self.film = torch.nn.Linear(in_channels, out_channels * 2)

        # The non-linear activation function.
        self.relu = torch.nn.ReLU(inplace=True)

        # The pooling layer used for down-sampling the output volume.
        self.pool = torch.nn.MaxPool2d(2)

        # The multiplicative (gamma) and additive (beta) linear transformation values.
        self.register_buffer("gamma", torch.zeros(out_channels, requires_grad=False))
        self.register_buffer("beta", torch.zeros(out_channels, requires_grad=False))

        # Field for controlling the current state of the layer.
        self.adapt = False

    def forward(self, x):

        # Computing a forward pass on the convolutional layer.
        z = self.bn(self.conv(x))

        # Computing feature wise linear modulation layer.
        if self.adapt is True:
            # Computing the average value for each channel in the volume.
            avg_channel = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))

            # Computing the gamma and beta weights and mean reducing in the batch dimension.
            self.gamma, self.beta = self.film(avg_channel.squeeze()).mean(dim=0).chunk(2)

        # Expanding tensor back into the correct dimension size.
        gamma = self.gamma[None, :, None, None].expand_as(z)
        beta = self.beta[None, :, None, None].expand_as(z)

        return self.pool(self.relu((1 + gamma) * z + beta))

    def initialize(self):
        torch.nn.init.normal_(self.conv.weight, 0, 0.01)
        torch.nn.init.normal_(self.film.weight, 0, 0.01)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def adapt_parameters(self):
        return self.conv.parameters()

    def film_parameters(self):
        return self.film.parameters()


# ============================================================
# Model Variants.
# ============================================================


class AdaConv32(_Conv):

    def __init__(self, **kwargs):
        super(AdaConv32, self).__init__(num_filters=32, **kwargs)


class AdaConv48(_Conv):

    def __init__(self, **kwargs):
        super(AdaConv48, self).__init__(num_filters=64, **kwargs)


class AdaConv64(_Conv):

    def __init__(self, **kwargs):
        super(AdaConv64, self).__init__(num_filters=64, **kwargs)


class AdaConv128(_Conv):

    def __init__(self, **kwargs):
        super(AdaConv128, self).__init__(num_filters=64, **kwargs)
