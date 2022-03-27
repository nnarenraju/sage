# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Mon Feb 21 16:36:39 2022

__author__      = nnarenraju
__copyright__   = Copyright 2021, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

def conv1D_layer_output(input_volume=None, kernel_size=None, padding=0,
                        stride=1):
    """
    torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, 
                    padding=0, dilation=1, groups=1, bias=True, 
                    padding_mode='zeros', device=None, dtype=None)
    """
    
    
    W = input_volume
    K = kernel_size
    P = padding
    S = stride
    output_dim = ((W-K+2.0*P)/S)+1
    return output_dim

def maxpool1D_layer_output(Lin=None, kernel_size=None, padding=0, dilation=1,
                           stride=None):
    """
    torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, 
                       return_indices=False, ceil_mode=False)
    """
    
    
    if stride == None:
        stride = kernel_size
        
    Lout = ((Lin + 2.0*padding - dilation * (kernel_size - 1) - 1)/stride) + 1
    return Lout


if __name__ == "__main__":
    
    # print(conv1D_layer_output(1669, 16))
    print(maxpool1D_layer_output(1654, 4))
