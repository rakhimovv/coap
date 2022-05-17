import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryEncoder(nn.Module):

    def __init__(self, in_dim=147, query_dim=128, hid_dim=256):
        super().__init__()
        # in_dim = 128 + k + 1 + 3
        self.lin0 = nn.Linear(in_dim, hid_dim)
        self.lin1 = nn.Linear(hid_dim, hid_dim - in_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.lin3 = nn.Linear(hid_dim, query_dim)

    def forward(self, x):
        y = F.softplus(self.lin0(x), beta=100)
        y = F.softplus(self.lin1(y), beta=100)
        y = F.softplus(self.lin2(torch.cat([x, y], dim=-1)), beta=100)
        y = F.softplus(self.lin3(y), beta=100)
        return y


class Decoder(nn.Module):

    def __init__(self, query_dim=128, hid_dim=256):
        super().__init__()
        in_dim = query_dim + 3
        self.lin0 = nn.Linear(in_dim, hid_dim)
        self.lin1 = nn.Linear(hid_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim - in_dim)
        self.lin3 = nn.Linear(hid_dim, hid_dim)
        self.lin4 = nn.Linear(hid_dim, hid_dim)
        self.lin5 = nn.Linear(hid_dim, hid_dim)
        self.lin6 = nn.Linear(hid_dim, 1)

    def forward(self, x):
        # x: query_dim + 3
        y = F.softplus(self.lin0(x), beta=100)
        y = F.softplus(self.lin1(y), beta=100)
        y = F.softplus(self.lin2(y), beta=100)
        y = F.softplus(self.lin3(torch.cat([x, y], dim=-1)), beta=100)
        y = F.softplus(self.lin4(y), beta=100)
        y = F.softplus(self.lin5(y), beta=100)
        y = self.lin6(y)
        return y


class ResnetPointnet(nn.Module):
    """ PointNet-based encoder network with ResNet blocks.

    Args:
        out_dim (int): dimension of latent code c
        hidden_dim (int): hidden dimension of the network
    """

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.out_dim = out_dim
        self.fc_pos = nn.Linear(in_dim, 2 * hidden_dim)
        self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, out_dim)

        self.act = nn.ReLU()

    @staticmethod
    def pool(x, dim=-1, keepdim=False):
        return x.max(dim=dim, keepdim=keepdim)[0]

    def forward(self, p):
        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.act(net))

        return c


class ResnetBlockFC(nn.Module):
    """ Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx
