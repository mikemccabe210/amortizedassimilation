import torch.nn as nn
import torch

class SpatialDropout1d(nn.Module):
    def __init__(self, p):
        super(SpatialDropout1d, self).__init__()
        self.p = p
        self.do = nn.Dropout2d(p)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.do(x)
        return x.squeeze(2)

class UNet(nn.Module):
    def __init__(self, in_dims, in_channels, out_channels, depth, initial_filters, upsampling = 'conv',
                 kernel =5, padding = 2, ptype = 'circular', nonlinearity = nn.SiLU, rate = .2,
                 gated = True):
        super(UNet, self).__init__()

        prev_filters = in_channels
        nfilters = initial_filters
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(nn.Sequential(downsampling_block(prev_filters, nfilters, kernel,
                                                     padding, ptype, nonlinearity, rate, gated),
                                                nn.LayerNorm([nfilters, in_dims//2**i])))
            prev_filters = nfilters
            nfilters *= 2

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            nfilters = prev_filters // 2
            self.up_path.append(nn.Sequential(upsampling_block(prev_filters, nfilters),
                                              nn.LayerNorm([nfilters, in_dims//2**i])))
            prev_filters = nfilters
        self.oxo = nn.Conv1d(prev_filters, out_channels, 1)

    def forward(self, x):
        outs = []
        for i, block in enumerate(self.down_path):
            x = block(x)
            if i != len(self.down_path)-1:
                outs.append(x)
        for block in self.up_path:
            x = block(x, outs.pop())
        return self.oxo(x)




class res_block(nn.Module):
    def __init__(self, n_in, filter_size = 16, kernel = 5, padding = 2, ptype = 'circular', nonlinearity = nn.SiLU, rate = 0,
                 stride = 1, gated = True):
        super(res_block, self).__init__()
        self.width, self.n_in = n_in[0], n_in[1]
        self.activation = nonlinearity
        self.gated = gated
        self.filter_size = filter_size

        self.block1 = nn.Sequential(nn.Conv1d(n_in, filter_size, kernel, stride, padding, padding_mode=ptype,
                                              bias = False),
                                    # nn.LayerNorm(self.width//2), #figure out
                                    nonlinearity,
                                    SpatialDropout1d(rate))
        fs = filter_size
        # If using gated units, double filter count
        if gated:
            fs *= 2
        self.block2 = nn.Sequential(nn.Conv1d(filter_size, fs, kernel, 1, padding, padding_mode = ptype,
                                               bias = False),
                                    # nn.LayerNorm(self.width // 2),
                                    nonlinearity,
                                    SpatialDropout1d(rate))
        self.down = nn.AvgPool1d(3, padding = 1, count_include_pad=False, stride = stride)
        # self.ln = nn.LayerNorm(self.width//stride)

    def forward(self, x):
        orig_x = x
        x = self.block1(x)
        x = self.block2(x)
        if self.gated:
            sig, lin = torch.split([self.filter_size, self.filter_size], 1)
            x = torch.sigmoid(sig) * lin
        print(orig_x, x)
        if x.shape[1] > orig_x.shape[1]:
            x[:, self.filter_size, :] += self.down(orig_x)
        else:
            x += orig_x
        return x

class downsampling_block(nn.Module):
    def __init__(self, n_in, filter_size, kernel = 5, padding = 2, ptype = 'circular', block_count = 1,
                 nonlinearity = nn.SiLU, rate=.2, gated = True):
        super(downsampling_block, self).__init__()
        self.downsample = res_block(n_in, filter_size, kernel, padding, ptype, nonlinearity, rate, 2, gated)
        self.blocks = nn.ModuleList()
        for i in range(block_count):
            self.blocks.append(res_block(n_in, filter_size, kernel, padding, ptype, nonlinearity, rate, 1, gated))
    def forward(self, x):
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x

class upsampling_block(nn.Module):
    def __init__(self, n_in, filter_size, kernel=5, padding=2, ptype='circular', block_count=1,
                 nonlinearity=nn.SiLU, rate=.2, gated=True):
        super(upsampling_block, self).__init__()
        self.upsample = nn.ConvTranspose1d(n_in, filter_size, kernel, 2, padding, padding_mode = ptype)
        self.blocks = nn.ModuleList()
        for i in range(block_count):
            self.blocks.append(res_block(n_in, filter_size, kernel, padding, ptype, nonlinearity, rate, 1, gated))

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], 1)
        for block in self.blocks:
            x = block(x)
        return x

