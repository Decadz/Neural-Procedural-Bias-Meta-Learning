import torch


class FiLMLinearBlock(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(FiLMLinearBlock, self).__init__()

        # The underlying linear layer (as implemented in PyTorch).
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)

        # Batch normalization layer to reduce internal covariance shift.
        self.bn = torch.nn.BatchNorm1d(out_features, track_running_stats=False)

        # The feature wise linear modulation (FiLM) layer for making the layer adaptive.
        self.film = torch.nn.Linear(in_features, out_features * 2)

        # The multiplicative (gamma) and additive (beta) linear transformation values.
        self.gamma = torch.zeros(out_features, requires_grad=False)
        self.beta = torch.zeros(out_features, requires_grad=False)

        # The non-linear activation function.
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x, adapt=False):
        # Computing a forward pass on the convolutional layer.
        z = self.bn(self.linear(x))

        # Computing feature wise linear modulation layer.
        if adapt is True:
            # Computing the average value for each channel in the volume.
            avg_channel = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))

            # Computing the gamma and beta weights and mean reducing in the batch dimension.
            self.gamma, self.beta = self.film(avg_channel.squeeze()).mean(dim=0).chunk(2)

        # Expanding tensor back into the correct dimension size.
        gamma = self.gamma[None, :].expand_as(z)
        beta = self.beta[None, :].expand_as(z)

        return self.relu((1 + gamma) * z + beta)

    def initialize(self):
        torch.nn.init.normal_(self.linear.weight, 0, 0.01)
        torch.nn.init.normal_(self.linear.weight, 0, 0.01)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def learn_parameters(self):
        return self.linear.parameters()

    def adapt_parameters(self):
        return self.film.parameters()