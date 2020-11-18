from utils import *
from cbr import *
from cbr_t import * 

class Decoder(nn.Module) :
  '''
  FuseNet Decoder implementation
  '''
  def __init__(self, in_ch , out_ch ) :
    '''
    Inputs :
      -in_ch : The size of the input channels
      -out_ch : The size of the output channels
    '''
    self.unpool_indexes = None

    super(Decoder, self).__init__()

    params = {'kernel_size' : 3, 'stride' : 1, 'padding' : 1}


    ''' Block 1 '''
    # Unpooling
    self.unpooling1 = nn.MaxUnpool2d((2,2), stride=2, padding = 0)
    # CBR
    self.cbr1 = CBR_T(512,512, params)
    self.cbr2 = CBR_T(512,512, params)
    self.cbr3 = CBR_T(512,256, params)
    # Dropout
    self.dropout1 = nn.Dropout2d(p=0.2)

    ''' End Bloc 1 '''

    ''' Block 2 '''
    # Unpooling
    self.unpooling2 = nn.MaxUnpool2d((2,2), stride=2, padding = 0 )
    # CBR
    self.cbr4 = CBR_T(256,256, params)
    self.cbr5 = CBR_T(256,256, params)
    self.cbr6 = CBR_T(256,128, params)
    # Dropout
    self.dropout2 = nn.Dropout2d(p=0.2)
    ''' End Bloc 2 '''

    ''' Block 3 '''
    # Unpooling
    self.unpooling3 = nn.MaxUnpool2d((2,2), stride=2, padding = 0 )
    # CBR
    self.cbr7 = CBR_T(128,128, params)
    self.cbr8 = CBR_T(128,64, params)
    # Dropout
    self.dropout3 = nn.Dropout2d(p=0.2)
    ''' End Bloc 3 '''

    ''' Block 4 '''
    # Unpooling
    self.unpooling4 = nn.MaxUnpool2d((2,2), stride=2, padding = 0 )
    # CBR
    self.cbr9 = CBR_T(64,64, params)
    self.cbr10 = CBR_T(64,3, params)

    ''' End Bloc 4 '''

    ''' Block 5 '''
    # Unpooling
    self.unpooling5 = nn.MaxUnpool2d((2,2), stride=2, padding = 0 )
    # CBR
    self.cbr11 = CBR_T(3,3, params )

    self.score = nn.Conv2d(3, out_ch, kernel_size = 1 )

    ''' End Bloc 5 '''


  def forward(self, x) :
    ''' Forward the deep neural netword for the decoder'''

    assert(len(self.unpool_indexes) == 5)

    # for m in self.unpool_indexes :
    #   print(m.shape)

    ''' Block 1 '''
    # Unpooling

    x = self.unpooling1(x, self.unpool_indexes[-1])

    # CBR
    x = self.cbr1(x)
    x = self.cbr2(x)
    x = self.cbr3(x)
    # Dropout
    x = self.dropout1(x)
    ''' End Block 1 '''

    ''' Block 2 '''
    # Unpooling

    x = self.unpooling2(x, self.unpool_indexes[-2])
    # CBR
    x = self.cbr4(x)
    x = self.cbr5(x)
    x = self.cbr6(x)
    # Dropout
    x = self.dropout2(x)

    ''' End Block 2 '''

    ''' Block 3 '''

    # Unpooling
    x = self.unpooling3(x, self.unpool_indexes[-3])
    # CBR
    x = self.cbr7(x)
    x = self.cbr8(x)
    # Dropout
    x = self.dropout3(x)

    ''' End Block 3 '''

    ''' Block 4 '''

    # Unpooling
    x = self.unpooling4(x, self.unpool_indexes[-4])
    # CBR
    x = self.cbr9(x)
    x = self.cbr10(x)

    ''' End Block 4 '''

    ''' Block 5 '''

    # Unpooling
    x = self.unpooling5(x, self.unpool_indexes[-5])
    # CBR
    x = self.cbr11(x)
    # Score
    x = self.score(x)
    ''' End Block 5 '''



    return x
