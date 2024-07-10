# Packages
import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveBlock(nn.Module):

    def __init__(self, num_layers, in_channels, num_filters, kernel_size, downsample=False):
        super().__init__()
        dilation_rates = [2**i for i in range(num_layers)]
        if downsample:
            if isinstance(downsample, bool):
                first_stride = 2
            else:
                first_stride = downsample
            first_kernel_size = first_stride + 1
            first_padding = (first_kernel_size-1)//2
            self.first_conv = nn.Conv1d(in_channels,
                                        num_filters,
                                        kernel_size=first_kernel_size,
                                        stride=first_stride,
                                        padding=first_padding)
        else:
            self.first_conv = nn.Conv1d(in_channels,
                                        num_filters,
                                        kernel_size=1,
                                        padding='same')
        self.tanh_conv = nn.ModuleList()
        self.sigm_conv = nn.ModuleList()
        self.final_conv = nn.ModuleList()
        for _, dilation_rate in enumerate(dilation_rates):
            self.tanh_conv.append(nn.Sequential(
                nn.Conv1d(num_filters,
                          num_filters,
                          kernel_size=kernel_size,
                          dilation=dilation_rate,
                          padding='same'),
                nn.Tanh(),
            ))
            self.sigm_conv.append(nn.Sequential(
                nn.Conv1d(num_filters,
                          num_filters,
                          kernel_size=kernel_size,
                          dilation=dilation_rate,
                          padding='same'),
                nn.Sigmoid(),
            ))
            self.final_conv.append(nn.Conv1d(num_filters,
                                             num_filters,
                                             kernel_size=1,
                                             padding='same'))
    
    def forward(self, x):
        x = self.first_conv(x)
        res_x = x
        for i in range(len(self.tanh_conv)):
            tanh_out = self.tanh_conv[i](x)
            sigm_out = self.sigm_conv[i](x)
            x = tanh_out * sigm_out
            x = self.final_conv[i](x)
            res_x = res_x + x
        
        return res_x


class WaveNet(nn.Module):

    def __init__(self,
                 in_channels: int = 2,
                 base_filters: int = 224,
                 wave_layers: tuple = (10, 6),
                 kernel_size: int = 3, 
                 downsample: int = 4,
                 sigmoid: bool = False, 
                 output_size: int = None, 
                 separate_channel: bool = False,
                 reinit: bool = True):

        super().__init__()

        wave_block = WaveBlock

        self.out_chans = len(wave_layers)
        self.out_size = output_size
        self.sigmoid = sigmoid
        self.separate_channel = separate_channel
        if isinstance(base_filters, int):
            base_filters = tuple([base_filters])

        self.spec_conv = nn.ModuleList()
        for i in range(self.out_chans):
            tmp_blocks = [wave_block(
                wave_layers[i], 
                in_channels, 
                base_filters[0], 
                kernel_size,
                downsample)]
            if len(base_filters) > 1:
                for j in range(len(base_filters) - 1):
                    tmp_blocks = tmp_blocks + [
                        wave_block(
                            wave_layers[i], 
                            base_filters[j], 
                            base_filters[j+1], 
                            kernel_size,
                            downsample)
                    ]
                self.spec_conv.append(nn.Sequential(*tmp_blocks))
            else:
                self.spec_conv.append(tmp_blocks[0])
        
        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))
        
        if reinit:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x): # x: (bs, ch, w)
        out = []
        for i in range(self.out_chans):
            out.append(self.spec_conv[i](x))
        out = torch.stack(out, dim=1)
        out = self.pool(out)
        return out
