import collections
import torch


class _AdaResNet(torch.nn.Module):

    def __init__(self, block_config, input_channels=3, num_ways=5, task_embedding_size=32, **kwargs):
        super(_AdaResNet, self).__init__()

        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("adapt1", _FiLMConvBlock(input_channels, block_config[0], task_embedding_size)),
            ("adapt2", _FiLMConvBlock(block_config[0], block_config[1], task_embedding_size)),
            ("adapt3", _FiLMConvBlock(block_config[1], block_config[2], task_embedding_size)),
            ("warp3", _FiLMWarpBlock(block_config[2], block_config[2], task_embedding_size)),
            ("adapt4", _FiLMConvBlock(block_config[2], block_config[3], task_embedding_size)),
            ("warp4", _FiLMWarpBlock(block_config[3], block_config[3], task_embedding_size)),
            ("adaPool", torch.nn.AdaptiveAvgPool2d(1)),
            ("flatten", torch.nn.Flatten())
        ]))

        # Creating the permutation invariant head of the network.
        self.classifier = _PermutationInvariantClassifier(block_config[-1], num_ways)

        # Model configurations and hyper-parameters.
        self.input_channels = input_channels
        self.block_config = block_config
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
            if isinstance(module, (_FiLMConvBlock, _FiLMWarpBlock)):
                module.initialize()

    def adapt_representation(self, task_embedding):

        # Resetting the batch norm running statistics.
        for module in self.modules():
            if hasattr(module, "reset_running_stats"):
                module.reset_running_stats()

        # Distributing the task_embedding to the encoder layers.
        for name, module in self.encoder.named_children():
            if isinstance(module, (_FiLMConvBlock, _FiLMWarpBlock)):
                module.adapt_representation(task_embedding)

        # Resetting the output layer using the output cone.
        self.classifier.adapt_representation(task_embedding)

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


class _FiLMConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, task_embedding_size):
        super(_FiLMConvBlock, self).__init__()

        # The convolutional feature extractor block, containing three filmed conv -> bn.
        self.block = torch.nn.Sequential(collections.OrderedDict([
            ("adapt1", _FiLMConv(in_channels, out_channels, task_embedding_size)),
            ("relu1", torch.nn.ReLU(inplace=True)),
            ("adapt2", _FiLMConv(out_channels, out_channels, task_embedding_size)),
            ("relu2", torch.nn.ReLU(inplace=True)),
            ("adapt3", _FiLMConv(out_channels, out_channels, task_embedding_size)),
        ]))

        # Residual (skip) connections. Cant been in sequential block since it runs in parallel.
        self.res_conv = torch.nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        self.res_bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)
        self.res_relu = torch.nn.ReLU(inplace=True)
        self.res_pool = torch.nn.MaxPool2d(2)

        # Block configurations and hyper-parameters.
        self.in_planes = in_channels
        self.planes = out_channels

    def forward(self, x):
        out = self.block(x)
        res = self.res_bn(self.res_conv(x))
        return self.res_pool(self.res_relu(out + res))

    def initialize(self):
        # Initializing the sequential block.
        for name, module in self.block.named_children():
            if isinstance(module, _FiLMConv):
                module.initialize()

        torch.nn.init.normal_(self.res_conv.weight, 0, 0.01)
        self.res_bn.weight.data.fill_(1)
        self.res_bn.bias.data.zero_()

    def meta_parameters(self):
        for module in self.block.children():
            if hasattr(module, "meta_parameters"):
                yield from module.meta_parameters()
        yield from self.res_conv.parameters()

    def base_parameters(self):
        for module in self.block.children():
            if hasattr(module, "base_parameters"):
                yield from module.base_parameters()
        yield from self.res_conv.parameters()


class _FiLMConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, task_embedding_size):
        super(_FiLMConv, self).__init__()

        # The underlying convolutional layer (as implemented in PyTorch).
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

        # Batch normalization layer to reduce internal covariance shift.
        self.bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)

        # The feature wise linear modulation (FiLM) layer for making the layer adaptive.
        self.film = torch.nn.Linear(task_embedding_size, out_channels * 2)

        # The multiplicative (gamma) and additive (beta) linear transformation values.
        self.register_buffer("gamma", torch.zeros(out_channels, requires_grad=False))
        self.register_buffer("beta", torch.zeros(out_channels, requires_grad=False))

    def forward(self, x):

        # Computing a forward pass on the convolutional layer.
        z = self.bn(self.conv(x))

        # Expanding the FiLM tensors back into the correct dimension size.
        gamma = self.gamma[None, :, None, None].expand_as(z)
        beta = self.beta[None, :, None, None].expand_as(z)

        # Applying the FiLM to the output.
        return (1 + gamma) * z + beta

    def initialize(self):
        torch.nn.init.normal_(self.conv.weight, 0, 0.01)
        torch.nn.init.normal_(self.film.weight, 0, 0.01)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def adapt_representation(self, task_embedding):
        # Computing the gamma and beta weights using the given task embedding.
        self.gamma, self.beta = self.film(task_embedding).chunk(2)

    def meta_parameters(self):
        yield from self.conv.parameters()
        yield from self.film.parameters()

    def base_parameters(self):
        yield from self.conv.parameters()


class _FiLMWarpBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, task_embedding_size):
        super(_FiLMWarpBlock, self).__init__()

        # The underlying convolutional layer (as implemented in PyTorch).
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

        # Batch normalization layer to reduce internal covariance shift.
        self.bn = torch.nn.BatchNorm2d(out_channels, track_running_stats=False)

        # The feature wise linear modulation (FiLM) layer for making the layer adaptive.
        self.film = torch.nn.Linear(task_embedding_size, out_channels * 2)

        # The multiplicative (gamma) and additive (beta) linear transformation values.
        self.register_buffer("gamma", torch.zeros(out_channels, requires_grad=False))
        self.register_buffer("beta", torch.zeros(out_channels, requires_grad=False))

    def forward(self, x):

        # Computing a forward pass on the convolutional layer.
        z = self.bn(self.conv(x))

        # Expanding tensor back into the correct dimension size.
        gamma = self.gamma[None, :, None, None].expand_as(z)
        beta = self.beta[None, :, None, None].expand_as(z)

        # Applying the FiLM to the output.
        return (1 + gamma) * z + beta

    def initialize(self):
        torch.nn.init.dirac_(self.conv.weight)
        torch.nn.init.normal_(self.film.weight, 0, 0.01)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def adapt_representation(self, task_embedding):
        # Computing the gamma and beta weights using the given task embedding.
        self.gamma, self.beta = self.film(task_embedding).chunk(2)

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

    def adapt_representation(self, task_embedding):
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


class AdaResNet(_AdaResNet):

    def __init__(self, **kwargs):
        super(AdaResNet, self).__init__(block_config=[64, 128, 256, 512], **kwargs)


class WideAdaResNet(_AdaResNet):

    def __init__(self, **kwargs):
        super(WideAdaResNet, self).__init__(block_config=[64, 160, 320, 640], **kwargs)
