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
                 kernel =3, padding = 1, ptype = 'circular', nonlinearity = nn.SiLU, rate = .2,
                 gated = True):
        super(UNet, self).__init__()

        nfilters = initial_filters
        self.init_layer = nn.Conv1d(in_channels, initial_filters//2, 1)
        prev_filters = initial_filters//2
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(downsampling_block(prev_filters, in_dims//2**(i+1), nfilters, kernel,
                                                     padding, ptype, 1, nonlinearity, rate, gated),
                                                )
            prev_filters = nfilters
            nfilters *= 2

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            nfilters = prev_filters // 2
            self.up_path.append(upsampling_block(prev_filters, in_dims//2**(i), nfilters, kernel,
                                                               padding, ptype, 1, nonlinearity, rate, gated),
                                              )
            prev_filters = nfilters
        self.oxo = nn.Conv1d(prev_filters, out_channels, 1)

    def forward(self, x):
        x = self.init_layer(x)
        outs = [x]
        for i, block in enumerate(self.down_path):
            x = block(x)
            if i != len(self.down_path)-1:
                outs.append(x)
            # print('down', x.shape)
        for block in self.up_path:
            skip = outs.pop()
            # print('up', x.shape, skip.shape)
            x = block((x, skip))
        return self.oxo(x)




class res_block(nn.Module):
    def __init__(self, n_in, filter_size = 16, kernel = 5, padding = 2, ptype = 'circular', nonlinearity = nn.SiLU, rate = 0,
                 stride = 1, gated = True, resample=True):
        super(res_block, self).__init__()
        self.n_in = n_in
        self.activation = nonlinearity
        self.gated = gated
        self.filter_size = filter_size
        self.resample = resample

        self.block1 = nn.Sequential(nn.Conv1d(n_in, filter_size, kernel, stride, padding, padding_mode=ptype,
                                              bias = True),
                                    nonlinearity(),
                                    SpatialDropout1d(rate))
        fs = filter_size
        # If using gated units, double filter count
        if gated:
            fs *= 2
        self.block2 = nn.Sequential(nn.Conv1d(filter_size, fs, kernel, 1, padding, padding_mode = ptype,
                                               bias = True),
                                    nonlinearity(),
                                    SpatialDropout1d(rate))
        if self.resample:
            self.down = nn.Conv1d(n_in, filter_size, kernel, stride=stride, padding=padding, padding_mode=ptype)

    def forward(self, x):
        orig_x = x
        x = self.block1(x)
        x = self.block2(x)
        if self.gated:
            sig, lin = torch.split(x, [self.filter_size, self.filter_size], 1)
            x = torch.sigmoid(sig) * lin
        if self.resample:
            # print(x.shape, self.down(orig_x).shape)
            return x + self.down(orig_x)
        else:
            return x + orig_x

class downsampling_block(nn.Module):
    def __init__(self, n_in, out_width, filter_size, kernel = 5, padding = 2, ptype = 'circular', block_count = 1,
                 nonlinearity = nn.SiLU, rate=.2, gated = True):
        super(downsampling_block, self).__init__()
        self.downsample = nn.Sequential(
                            res_block(n_in, filter_size, kernel, padding, ptype, nonlinearity, rate, 2, gated),
                            nn.LayerNorm([filter_size, out_width])
                            )
        self.blocks = nn.ModuleList()
        for i in range(block_count):
            self.blocks.append(nn.Sequential(
                res_block(filter_size, filter_size, kernel, padding, ptype, nonlinearity, rate, 1, gated),
                nn.LayerNorm([filter_size, out_width])))
    def forward(self, x):
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x

class upsampling_block(nn.Module):
    def __init__(self, n_in, out_width, filter_size, kernel=5, padding=2, ptype='circular', block_count=1,
                 nonlinearity=nn.SiLU, rate=.2, gated=True):
        super(upsampling_block, self).__init__()
        self.upsample = nn.ConvTranspose1d(n_in, filter_size, kernel, 2, padding, output_padding=1)
        self.blocks = nn.ModuleList()
        for i in range(block_count):
            self.blocks.append(nn.Sequential(
                res_block(n_in, filter_size, kernel, padding, ptype, nonlinearity, rate, 1, gated),
            nn.LayerNorm([filter_size, out_width])))

    def forward(self, x):
        x, skip = x
        x = self.upsample(x)
        x = torch.cat([x, skip], 1)
        # x = self.nin(x)
        for block in self.blocks:
            x = block(x)
        return x

