"""
Original Repository: https://github.com/hfawaz/InceptionTime
Reference paper: https://arxiv.org/abs/1909.04939

Modified to PyTorch: Narenraju Nagarajan (19/01/2024)

"""

# resnet model
import torch
import torch.nn as nn


class InceptionTime(nn.Module):

    def __init__(self, num_classes, in_channels=2, num_kernels=3, nfilters=32, 
                 use_residual=True, use_bottleneck=True, 
                 depth=6, kernel_size=41,
                 weights_reinit=True):

        super(InceptionTime, self).__init__()

        self.nfilters = nfilters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.bottleneck_size = 32
        self.stride = 1
        self.num_classes = num_classes

        ## Generic layers and params
        self.relu = nn.ReLU()
        _kwargs = {'padding': 'same', 'bias': False}

        """ Inception Module """
        # Different kernel sizes for varied receptive fields
        kernel_sizes = [self.kernel_size // (2 ** i) for i in range(num_kernels)]

        ## Make a module for each depth
        _in_channels = in_channels
        self.inception_input = nn.ModuleDict({})
        self.varied_kernels = nn.ModuleDict({})
        self.inception_output = nn.ModuleDict({})
        self.inmod_bnorm1d = nn.ModuleDict({})
        for d in range(self.depth):
            ## Step 0: Inception input
            inlayer = nn.Conv1d(in_channels=_in_channels, out_channels=self.bottleneck_size, kernel_size=1, **_kwargs)
            self.inception_input.update(nn.ModuleDict({str(d): inlayer}))
            ## Step 1: Varied kernel sizes
            tmp1 = []
            for n in range(len(kernel_sizes)):
                tmp1.append(nn.Conv1d(in_channels=self.bottleneck_size, out_channels=self.nfilters, 
                                      kernel_size=kernel_sizes[n], stride=self.stride, 
                                      **_kwargs))
            self.varied_kernels.update(nn.ModuleDict({str(d): nn.ModuleList(tmp1)}))
            ## Step 2: Inception output
            # Refer link: https://stackoverflow.com/questions/71021725/how-to-use-same-padding-for-maxpool1d
            # padding == 'same' in Keras can be implemented by setting padding to (kernel_size - 1)/2
            # for uneven kernels.
            tmp2 = []
            tmp2.append(nn.MaxPool1d(kernel_size=3, stride=self.stride, padding=int((3 - 1)/2)))
            tmp2.append(nn.Conv1d(in_channels=_in_channels, out_channels=self.nfilters, kernel_size=1, **_kwargs))
            self.inception_output.update(nn.ModuleDict({str(d): nn.ModuleList(tmp2)}))
            ## Step 3: Update _in_channels
            _in_channels = int((num_kernels+1) * self.nfilters)

            # Inception batchnorm1d
            self.inmod_bnorm1d.update(nn.ModuleDict({str(d): nn.BatchNorm1d(_in_channels)}))

        """ Shortcut Module """
        self.shortchut_module = nn.ModuleDict({})
        self.shmod_bnorm1d = nn.ModuleDict({})
        _in_channels = in_channels
        for d in range(self.depth):
            if d%3==2:
                shmod = nn.Conv1d(in_channels=_in_channels, out_channels=int((num_kernels+1) * self.nfilters), 
                                  kernel_size=1, **_kwargs)
                self.shortchut_module.update(nn.ModuleDict({str(d): shmod}))
                _in_channels = int((num_kernels+1) * self.nfilters)
                self.shmod_bnorm1d.update(nn.ModuleDict({str(d): nn.BatchNorm1d(_in_channels)}))

        """ Output layers """
        # NOTE: Softmax activation is removed favouring the ORChiD backend instead
        self.output_layer = nn.Linear(4254, 512)

        ### Weight Re-initialisation
        if weights_reinit:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)


    def _inception_module(self, input_tensor, d):
        # Step 0: Bottleneck
        if self.use_bottleneck and int(input_tensor.shape[1]) > 1:
            input_inception = self.inception_input[d](input_tensor)
        else:
            input_inception = input_tensor
        # Step 1: 
        inmod = []
        for layer in self.varied_kernels[d]:
            inmod.append(layer(input_inception))
        # Step 2: 
        maxpool = self.inception_output[d][0](input_tensor)
        inmod.append(self.inception_output[d][1](maxpool))
        # Step 3: 
        x = torch.cat(inmod, dim=1)
        x = self.inmod_bnorm1d[d](x)
        x = self.relu(x)
        return x


    def _shortcut_layer(self, input_tensor, out_tensor, d):
        shortcut_y = self.shortchut_module[d](input_tensor)
        shortcut_y = self.shmod_bnorm1d[d](shortcut_y)
        x = torch.add(shortcut_y, out_tensor)
        x = self.relu(x)
        return x


    def forward(self, input_tensor):
        # Save copy of input for skip connections
        x = input_tensor.detach().clone()
        input_res = input_tensor.detach().clone()
        # Inception layer and skip connections
        for d in range(self.depth):
            x = self._inception_module(x, str(d))
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x, str(d))
                input_res = x
        
        # GlobalAveragePooling (GAP) 1D
        # Refer to the Keras link below for implementation
        # https://github.com/keras-team/keras/blob/v3.0.2/keras/layers/pooling/global_average_pooling1d.py#L7
        # Link to Pytorch forum regarding the issue
        # https://discuss.pytorch.org/t/how-can-i-perform-global-average-pooling-before-the-last-fully-connected-layer/74352
        # steps_axis = 1 if self.data_format == "channels_last" else 2
        steps_axis = 1
        gap_layer = torch.mean(x, dim=steps_axis)

        # Final linear layer to reduce tensor to number of classes for classification
        out = self.output_layer(gap_layer)
        return out


def inceptiontime_def(num_classes=512):
    """ Constructs a Default InceptionTime model """
    model = InceptionTime(num_classes, nfilters=32, use_residual=True, use_bottleneck=True, 
                          depth=6, kernel_size=41, weights_reinit=False)
    return model