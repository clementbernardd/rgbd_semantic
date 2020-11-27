from utils import *
from cbr import *
from cbr_t import *
from gcn_module import * 

class GCN_DECODER(nn.Module) :
  ''' Decoder of the GCN '''
  def __init__(self,in_ch, out_ch) :
    '''
    Inputs :
      -in_ch : The size of the input channels
      -out_ch : The size of the output channels
    '''
    self.unpool_indexes = None
    self.pool = None
    self.gcns = None

    super(GCN_DECODER, self).__init__()

    params = {'kernel_size' : (3, 3) , 'stride' : (1, 1), 'padding' : (1, 1)}
    params_pool = {'kernel_size':2, 'stride':2, 'padding':0}

    ''' Block 1 '''
    # Unpooling
    self.unpooling1 = nn.MaxUnpool2d(**params_pool)
    # CBR
    self.cbr1 = CBR_T(512,512, params)
    self.cbr2 = CBR_T(512,512, params)
    self.cbr3 = CBR_T(512,512, params)
    ''' End Block 1  '''

    ''' Block 2 '''
    # Unpooling
    self.unpooling2 = nn.MaxUnpool2d(**params_pool )
    # CBR
    self.cbr4 = CBR_T(512,512, params)
    self.cbr5 = CBR_T(512,512, params)
    self.cbr6 = CBR_T(512,256, params)
    ''' End Block 2 '''

    ''' Block 3 '''
    # Unpooling
    self.unpooling3 = nn.MaxUnpool2d(**params_pool )
    # CBR
    self.cbr7 = CBR_T(256,256, params)
    self.cbr8 = CBR_T(256,128, params)
    ''' End Block 3 '''


    ''' Block 4 '''
    # Unpooling
    self.unpooling4 = nn.MaxUnpool2d(**params_pool )
    # CBR
    self.cbr9 = CBR_T(128,128, params)
    self.cbr10 = CBR_T(128,64, params)

    ''' End Block 4 '''

    ''' Block 5 '''
    # Unpooling
    self.unpooling5 = nn.MaxUnpool2d(**params_pool)
    # CBR
    self.cbr11 = CBR_T(64,3, params )

    self.score = nn.Conv2d(3, out_ch, kernel_size = 1 )

    ''' End Block 5 '''


  def forward(self ,x ) :

    # Assert that the unpooling indexes are not None
    assert(  (self.unpool_indexes is not None) and (self.pool is not None )  and (self.gcns is not None)      )

    ''' Block 1 '''
    # Unpooling
    x = self.unpooling1(x, self.unpool_indexes[-1], output_size=self.pool[-1])
    # GCN
    x = self.gcns[-1] + x
    # CBR
    x = self.cbr1(x)
    x = self.cbr2(x)
    x = self.cbr3(x)
    ''' End Block 1 '''

    ''' Block 2 '''
    # Unpooling
    x = self.unpooling2(x, self.unpool_indexes[-2], output_size=self.pool[-2])
    # GCN
    x = self.gcns[-2] + x
    # CBR
    x = self.cbr4(x)
    x = self.cbr5(x)
    x = self.cbr6(x)

    ''' End Block 2 '''

    ''' Block 3 '''

    # Unpooling
    x = self.unpooling3(x, self.unpool_indexes[-3], output_size=self.pool[-3])
    # GCN
    x = self.gcns[-3] + x
    # CBR
    x = self.cbr7(x)
    x = self.cbr8(x)

    ''' End Block 3 '''

    ''' Block 4 '''

    # Unpooling
    x = self.unpooling4(x, self.unpool_indexes[-4], output_size=self.pool[-4])
    # GCN
    x = self.gcns[-4] + x
    # CBR
    x = self.cbr9(x)
    x = self.cbr10(x)

    ''' End Block 4 '''

    ''' Block 5 '''

    # Unpooling
    x = self.unpooling5(x, self.unpool_indexes[-5], output_size=self.pool[-5])
    # CBR
    x = self.cbr11(x)
    # Score
    x = self.score(x)
    ''' End Block 5 '''

    return x
