from utils import *
from cbr import *
from gcn_module import * 

class GCN_ENCODER(nn.Module) :
  ''' Encoder of the GCN network '''
  def __init__(self, in_rgb_ch = 3, in_depth_ch = 1 , k = 5 ) :
    '''
      Inputs :
        -in_rgb_ch : Input of the RGB channel
        -in_depth_ch : Input of the depth channel
        -k : The size of the convolution kernel
    '''
    super(GCN_ENCODER, self).__init__()

    params = {'kernel_size' : (3, 3) , 'stride' : (1, 1), 'padding' : (1, 1)}
    params_pool = {'kernel_size':2, 'stride':2, 'padding':0, 'dilation':1, 'ceil_mode':False, 'return_indices' : True}

    ''' BLOCK 1 '''

    self.cbr1 = CBR(in_rgb_ch, 64 ,params )
    self.cbr2 = CBR(64, 64, params)

    self.cbr1_ = CBR(in_depth_ch, 64 , params)
    self.cbr2_ = CBR(64, 64, params)

    self.pooling1 = nn.MaxPool2d(**params_pool)

    ''' END BLOCK 1 '''

    ''' BLOCK 2 '''

    self.cbr3 = CBR(64, 128, params)
    self.cbr4 = CBR(128,128, params)

    self.cbr3_ = CBR(64,128 , params)
    self.cbr4_ = CBR(128, 128, params)

    self.pooling2 = nn.MaxPool2d(**params_pool)

    self.gcn1 = GCN_MODULE(128, 128, 5 )

    ''' END BLOCK 2 '''

    ''' BLOCK 3 '''

    self.cbr5 = CBR(128, 256, params)
    self.cbr6 = CBR(256, 256, params)
    self.cbr7 = CBR(256, 256, params)

    self.pooling3 = nn.MaxPool2d(**params_pool)

    self.cbr5_ = CBR(128, 256, params)
    self.cbr6_ = CBR(256, 256, params)
    self.cbr7_ = CBR(256, 256, params)

    self.gcn2 = GCN_MODULE(256, 256, 5 )

    ''' END BLOCK 3 '''

    ''' BLOCK 4 '''

    self.cbr8 = CBR( 256,512, params)
    self.cbr9 = CBR(512, 512, params)
    self.cbr10 = CBR(512, 512, params)

    self.pooling4 = nn.MaxPool2d(**params_pool)

    self.cbr8_ = CBR( 256,512, params)
    self.cbr9_ = CBR(512, 512, params)
    self.cbr10_ = CBR(512, 512, params)

    self.gcn3 = GCN_MODULE(512, 512, 5 )

    ''' END BLOCK 4 '''

    ''' BLOCK 5 '''

    self.cbr11 = CBR( 512,512, params)
    self.cbr12 = CBR(512, 512, params)
    self.cbr13 = CBR(512, 512, params)


    self.pooling5 = nn.MaxPool2d(**params_pool )

    self.cbr11_ = CBR( 512,512, params)
    self.cbr12_ = CBR(512, 512, params)
    self.cbr13_ = CBR(512, 512, params)

    self.gcn4 = GCN_MODULE(512, 512, 5 )

    ''' END BLOCK 5 '''

  def forward(self,x) :

    rgb_encoder = x[0]
    depth_encoder = x[1]
    # Unpool indices
    unpool = []
    # Pool indices
    pool = []
    # The gcn outputs
    gcns = []
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
    # GCN
    gcn = self.gcn1(fusion)
    gcns.append(gcn)
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
    # GCN
    gcn = self.gcn2(fusion)
    gcns.append(gcn)
    # Pooling
    pool.append(rgb_encoder.shape)
    rgb_encoder, indices = self.pooling3(fusion)
    depth_encoder ,_= self.pooling3(depth_encoder)
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
    # GCN
    gcn = self.gcn3(fusion)
    gcns.append(gcn)
    # Pooling
    pool.append(rgb_encoder.shape)
    rgb_encoder, indices = self.pooling4(fusion)
    depth_encoder ,_ = self.pooling4(depth_encoder)
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
    # GCN
    gcn = self.gcn4(fusion)
    gcns.append(gcn)
    # Pooling
    pool.append(rgb_encoder.shape)
    rgb_encoder, indices = self.pooling4(fusion)
    # Keep the indices for the unpooling operation
    unpool.append(indices)

    ''' End Block 5 '''

    return rgb_encoder, unpool, pool, gcns
