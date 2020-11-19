from utils import *
from cbr import *

class Encoder(nn.Module) :
  '''
      FuseNet encoder implementation
      The CONV+POOL parameters are the ones of the VGG-16 Net as referred in the article
  '''
  def __init__(self, in_rgb_ch , in_depth_ch ) :

    super(Encoder, self).__init__()

    param_0 = {'kernel_size' : 1, 'stride' : 1, 'padding' : 0}
    params = {'kernel_size' : 3, 'stride' : 1, 'padding' : 1}

    ''' Block 1'''
    self.cbr1 = CBR(in_rgb_ch, 3 , param_0)
    self.cbr2 = CBR(3, 3, param_0)

    self.cbr1_ = CBR(in_depth_ch, 3 , param_0)
    self.cbr2_ = CBR(3, 3, param_0)

    self.pooling1 = nn.MaxPool2d((2,2), stride = 2, return_indices = True )

    ''' End Block 1'''

    ''' Block 2'''

    self.cbr3 = CBR(3, 64, params)
    self.cbr4 = CBR(64,64, params)

    self.cbr3_ = CBR(3,64 , params)
    self.cbr4_ = CBR(64, 64, params)

    self.pooling2 = nn.MaxPool2d((2,2), stride = 2 , return_indices = True )

    ''' End Block 2'''

    ''' Block 3'''

    self.cbr5 = CBR(64, 128, params)
    self.cbr6 = CBR(128, 128, params)
    self.cbr7 = CBR(128, 128, params)

    self.pooling3 = nn.MaxPool2d((2,2), stride = 2 , return_indices = True )
    self.dropout1 = nn.Dropout2d(p=0.2)

    self.cbr5_ = CBR(64, 128, params)
    self.cbr6_ = CBR(128, 128, params)
    self.cbr7_ = CBR(128, 128, params)

    self.dropout1_ = nn.Dropout2d(p=0.2)

    ''' End Block 3'''

    ''' Block 4 '''
    self.cbr8 = CBR( 128,256, params)
    self.cbr9 = CBR(256, 256, params)
    self.cbr10 = CBR(256, 256, params)

    self.pooling4 = nn.MaxPool2d((2,2), stride = 2 , return_indices = True )
    self.dropout2 = nn.Dropout2d(p=0.2)

    self.cbr8_ = CBR( 128,256, params)
    self.cbr9_ = CBR(256, 256, params)
    self.cbr10_ = CBR(256, 256, params)

    self.dropout2_ = nn.Dropout2d(p=0.2)

    ''' End Block 4'''

    ''' Block 5'''
    self.cbr11 = CBR( 256,512, params)
    self.cbr12 = CBR(512, 512, params)
    self.cbr13 = CBR(512, 512, params)


    self.pooling5 = nn.MaxPool2d((2,2), stride = 2 ,padding = 0 , return_indices = True )
    self.dropout3 = nn.Dropout2d(p=0.2)

    self.cbr11_ = CBR( 256,512, params)
    self.cbr12_ = CBR(512, 512, params)
    self.cbr13_ = CBR(512, 512, params)
    ''' End Block 5'''


  def forward(self,x):
    ''' x is the RGB image and the depth '''

    rgb_encoder = x[0]
    depth_encoder = x[1]
    # Unpool indices
    unpool = []
    # Pool indices
    pool = []

    ''' Block 1'''
    # CBR RGB
    rgb_encoder = self.cbr1(rgb_encoder)
    rgb_encoder = self.cbr2(rgb_encoder)
    # CBR Depth
    depth_encoder = self.cbr1_(depth_encoder)
    depth_encoder = self.cbr2_(depth_encoder)
    # Fusion
    fusion = depth_encoder + rgb_encoder
    # Pooling
    pool.append(rgb_encoder.shape)
    rgb_encoder, indices = self.pooling1(fusion)
    depth_encoder, _ = self.pooling1(depth_encoder)
    # Keep the indices for the unpooling operation
    unpool.append(indices)

    ''' End Block 1 '''

    ''' Block 2 '''
    # CBR RGB
    rgb_encoder = self.cbr3(rgb_encoder)
    rgb_encoder = self.cbr4(rgb_encoder)
    # CBR Depth
    depth_encoder = self.cbr3_(depth_encoder)
    depth_encoder = self.cbr4_(depth_encoder)
    # Fusion
    fusion = depth_encoder + rgb_encoder
    # Pooling
    pool.append(rgb_encoder.shape)
    rgb_encoder, indices = self.pooling2(fusion)
    depth_encoder , _ = self.pooling2(depth_encoder)

    # Keep the indices for the unpooling operation
    unpool.append(indices)

    ''' End Block 2 '''

    ''' Block 3 '''

    # CBR RGB
    rgb_encoder = self.cbr5(rgb_encoder)
    rgb_encoder = self.cbr6(rgb_encoder)
    rgb_encoder = self.cbr7(rgb_encoder)
    # CBR Depth
    depth_encoder = self.cbr5_(depth_encoder)
    depth_encoder = self.cbr6_(depth_encoder)
    depth_encoder = self.cbr7_(depth_encoder)
    # Fusion
    fusion = depth_encoder + rgb_encoder
    # Pooling
    pool.append(rgb_encoder.shape)
    rgb_encoder, indices = self.pooling3(fusion)
    depth_encoder ,_= self.pooling3(depth_encoder)
    # Dropout
    rgb_encoder = self.dropout1(rgb_encoder)
    depth_encoder = self.dropout1_(depth_encoder)
    # Keep the indices for the unpooling operation
    unpool.append(indices)

    ''' End Block 3 '''

    ''' Block 4 '''

    # CBR RGB
    rgb_encoder = self.cbr8(rgb_encoder)
    rgb_encoder = self.cbr9(rgb_encoder)
    rgb_encoder = self.cbr10(rgb_encoder)
    # CBR Depth
    depth_encoder = self.cbr8_(depth_encoder)
    depth_encoder = self.cbr9_(depth_encoder)
    depth_encoder = self.cbr10_(depth_encoder)
    # Fusion
    fusion = depth_encoder + rgb_encoder
    # Pooling
    pool.append(rgb_encoder.shape)
    rgb_encoder, indices = self.pooling4(fusion)
    depth_encoder ,_ = self.pooling4(depth_encoder)
    # Dropout
    rgb_encoder = self.dropout2(rgb_encoder)
    depth_encoder = self.dropout2_(depth_encoder)
    # Keep the indices for the unpooling operation
    unpool.append(indices)

    ''' End Block 4 '''

    ''' Block 5 '''
    # CBR RGB
    rgb_encoder = self.cbr11(rgb_encoder)
    rgb_encoder = self.cbr12(rgb_encoder)
    rgb_encoder = self.cbr13(rgb_encoder)
    # CBR Depth
    depth_encoder = self.cbr11_(depth_encoder)
    depth_encoder = self.cbr12_(depth_encoder)
    depth_encoder = self.cbr13_(depth_encoder)
    # Fusion
    fusion = depth_encoder + rgb_encoder
    # Pooling
    pool.append(rgb_encoder.shape)
    rgb_encoder, indices = self.pooling4(fusion)
   # Dropout
    rgb_encoder = self.dropout3(rgb_encoder)
    # Keep the indices for the unpooling operation
    unpool.append(indices)

    ''' End Block 5 '''

    return rgb_encoder, unpool, pool
