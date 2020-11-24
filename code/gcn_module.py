from utils import *

class GCN_MODULE(nn.Module) :
  ''' Class that does the GCN : Global Convolutional Network '''
  def __init__(self, in_ch , out_ch , k ) :
    '''
      Inputs :
        -in_ch : Input channel size
        -out_ch : Output channel size
        -k : The size of the convolution kernel
    '''
    super(GCN_MODULE, self).__init__()

    params_kernel_1 = {'kernel_size' : (k,1), 'stride' : 1, 'padding' : 1}
    params_kernel_2 = {'kernel_size' : (1,k), 'stride' : 1, 'padding' : 1}

    # Convolution of size (k,1)
    self.conv1 = nn.Conv2d(in_ch, out_ch, **params_kernel_1)
    self.conv2 = nn.Conv2d(out_ch,out_ch, **params_kernel_2)

    # Convolution of size (1,k)
    self.conv1_ = nn.Conv2d(in_ch, out_ch, **params_kernel_2)
    self.conv2_ = nn.Conv2d(out_ch,out_ch, **params_kernel_1)

  def forward(self, x) :
    # Split to feed the 2 convolutions
    y = x
    # Convolution of size (k,1)
    x = self.conv1(x)
    x = self.conv2(x)
    # Convolution of size (1,k)
    y = self.conv1_(y)
    y = self.conv2_(y)
    # Sum the 2 convolution layers
    z = x + y
    return z
