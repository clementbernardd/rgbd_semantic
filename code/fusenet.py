from utils import *
from encoder import *
from decoder import *

class FuseNet(nn.Module) :
    ''' FuseNet implementation '''
    def __init__(self, N,  name = 'fusenet' ) :
        '''
            Inputs :
                - N : The number of classes in the dataset
                - name : The name of the model for the saving processing
        '''
        super(FuseNet, self).__init__()
        self.N = N
        self.name = name
        self.checkpoint_file = os.path.join('../model/' , name )

        self.encoder = Encoder(3,1)
        self.decoder = Decoder(512, N)


    def forward(self, x):
        # Encode the data
        x, unpool, pool = self.encoder(x)
        # Add the unpool indexes
        self.decoder.unpool_indexes = unpool
        self.decoder.pool = pool
        # Decode the data
        x = self.decoder(x)
        # Argmax for the prediction
        x = torch.nn.functional.log_softmax(x, dim = 1 )

        return x.type(torch.float64)

    def save_checkpoint(self) :
        print('--- Save model checkpoint ---')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self) :
        print('--- Loading model checkpoint ---')
        if torch.cuda.is_available() :

            self.load_state_dict(torch.load(self.checkpoint_file))

        else :
            self.load_state_dict(torch.load(self.checkpoint_file,map_location=torch.device('cpu')))
