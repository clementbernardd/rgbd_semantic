from utils import *
from encoder import *
from decoder import * 

class FuseNet(nn.Module) :
    ''' FuseNet implementation '''
    def __init__(self, N) :
        '''
            Inputs :
                - N : The number of classes in the dataset
        '''
        super(FuseNet, self).__init__()
        self.encoder = Encoder(3,1)
        self.decoder = Decoder(512, N)


    def forward(self, x):
        # Encode the data
        x, unpool = self.encoder(x)
        # Add the unpool indexes
        self.decoder.unpool_indexes = unpool
        # Decode the data
        x = self.decoder(x)
        # Argmax for the prediction
        x = torch.softmax(x, dim=1)

        return x.type(torch.float64)
