from utils import * 

class CBR_T(nn.Module) :
    ''' Class that does the Transposed convolution + BN + ReLU '''
    def __init__(self, in_ch , out_ch, params = None ) -> None  :
        '''
        Inputs :
            - in_ch : The size of the input
            - out_ch : The size of the output
            - params : The parameters of the convolution
        '''
        super(CBR_T, self).__init__()

        if params is None :
          params = {'kernel_size' :  3, 'stride': 1 , 'padding' : 0}

        self.cbr = nn.Sequential(
            # Convolution
            nn.ConvTranspose2d(in_ch, out_ch, **params ),
            # Batch normalisation
            nn.BatchNorm2d(out_ch),
            # ReLU
            nn.ReLU(True)
        )

    def forward(self, x) :

        x = self.cbr(x)

        return x
