import collections
import torch


class _RelationNetwork(torch.nn.Module):

    def __init__(self, input_channels, num_ways, num_shots, test_shots, block_config, **kwargs):
        super(_RelationNetwork, self).__init__()

        # The encoder network for preprocessing the instances into embeddings.
        self.encoder = torch.nn.Sequential(collections.OrderedDict([
            ("block1", _ConvBlock(input_channels, block_config[0], downsample=True)),
            ("block2", _ConvBlock(block_config[0], block_config[1], downsample=True)),
            ("block3", _ConvBlock(block_config[1], block_config[2], downsample=False)),
            ("block4", _ConvBlock(block_config[2], block_config[3], downsample=False)),
        ]))

        # The relation network for relating support to query instances.
        self.relation = torch.nn.Sequential(collections.OrderedDict([
            ("block1", _ConvBlock(2 * block_config[3], block_config[3], downsample=True)),
            ("block2", _ConvBlock(block_config[3], block_config[3], downsample=True)),
            ("adaPool", torch.nn.AdaptiveAvgPool2d(1)),
            ("flatten", torch.nn.Flatten()),
            ("linear", torch.nn.Linear(block_config[-1], 1)),
            ("sigmoid", torch.nn.Sigmoid())
        ]))

        # Model configuration hyper-parameters.
        self.block_config = block_config
        self.input_channels = input_channels

        # Few-shot learning settings.
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.test_shots = test_shots

        # Initializing the model's parameters.
        self.initialize()

    def forward(self, x):

        # Generating embeddings for each instances.
        z = self.encoder(x)

        # Partitioning out the support and query embeddings.
        z_support, z_query = torch.split(z, [self.num_shots * self.num_ways, self.test_shots * self.num_ways], dim=0)

        # Partitioning out the support embeddings into chunks of size equal to the number of ways.
        z_support_ways = torch.chunk(z_support, chunks=self.num_ways, dim=0)

        # A list for recording the relation scores.
        relation_embeddings = []

        # Generating each relation in the output.
        for z_support_i in z_support_ways:

            # Summing all the support embeddings of the same class into a single embedding.
            z_support_i = torch.sum(z_support_i, dim=0)

            # Adding a new dimension to match the tensor size of z_query.
            z_support_i = z_support_i.unsqueeze(0).expand(self.test_shots * self.num_ways, -1, -1, -1)

            # Concatenate along the second dimension (axis=1)
            z_support_query_i = torch.cat((z_support_i, z_query), dim=1)

            # Computing the final relation scores for the current way.
            y_i = self.relation(z_support_query_i)
            relation_embeddings.append(y_i)

        # Returning the final relation scores.
        return torch.cat(relation_embeddings, dim=1)

    def initialize(self):
        # Initializing the networks parameters.
        for name, module in self.encoder.named_children():
            if isinstance(module, _ConvBlock):
                module.initialize()
        for name, module in self.relation.named_children():
            if isinstance(module, _ConvBlock):
                module.initialize()

# ============================================================
# Residual Network block definition.
# ============================================================


class _ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, downsample):
        super(_ConvBlock, self).__init__()

        # The convolutional feature extractor block, containing three filmed conv -> bn.
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
        self.pool = torch.nn.MaxPool2d(2 if downsample else 1)

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

# ============================================================
# Model Variants.
# ============================================================


class RelationNetwork(_RelationNetwork):

    def __init__(self, **kwargs):
        super(RelationNetwork, self).__init__(block_config=[64, 128, 256, 512], **kwargs)


class WideRelationNetwork(_RelationNetwork):

    def __init__(self, **kwargs):
        super(WideRelationNetwork, self).__init__(block_config=[64, 160, 320, 640], **kwargs)
