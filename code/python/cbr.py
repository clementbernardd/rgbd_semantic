from utils import *


class CBR(nn.Module) :
    ''' Class that does the Conv + BN + ReLU '''
    def __init__(self, in_ch , out_ch, params = None ) -> None  :
        '''
        Inputs :
            - in_ch : The size of the input
            - out_ch : The size of the output
            - params : The parameters of the convolution
        '''
        super(CBR, self).__init__()

        if params is None :
          params = {'kernel_size' :  3, 'stride': 1 , 'padding' : 0}

        params_batch = {'eps':1e-05, 'momentum':0.1, 'affine':True, 'track_running_stats':True}

        self.cbr = nn.Sequential(
            # Convolution
            nn.Conv2d(in_ch, out_ch, **params ),
            # Batch normalisation
            nn.BatchNorm2d(out_ch, **params_batch),
            # ReLU
            nn.ReLU(True)
        )

    def forward(self, x) :

        x = self.cbr(x)

        return x
