''' Importations '''

from utils import *
from importations import *
from fusenet import *
from gcn import *
from evaluation import *

''' Class that implements the training process '''

class Train(object) :
    ''' Object that deals the training process '''
    def __init__(self, X_train, y_train ,BATCH_SIZE, print_iter, criterion, \
                 optimizer, learning_rate, model,N, name, X_val = None, y_val = None  ) :
        '''
        Inputs :
            - X_train : The training inputs
            - y_train : The training labels
            - BATCH_SIZE : The size of the mini-batch used for the training process
            - print_iter : The number of time we plot the loss/scores during the training
            - criterion : The criterion used
            - optimizer : The optimizer used (like : optim.Adam)
            - learning_rate : The learning rate for the optimizer
            - model : FuseNet or GCN
            - N : the number of classes
            - name : The name where will be stored the model parameters
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.BATCH_SIZE = BATCH_SIZE
        self.print_iter = print_iter
        self.criterion = criterion
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model(N+1, name).to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.model.parameters(), lr = learning_rate)
        self.name = name
        self.N = N
        self.X_val = X_val
        self.y_val = y_val


        self.evaluation = Evaluation()
        self.chkpt_dir = '../../tmp/'+ self.name

        self.scores_train = {}
        self.scores_val = {}
        self.loss_train = {}


    def predict(self, X ) :
        ''' Predict the output for X as inputs '''
        # Check if the size of the batch is below the BATCH_SIZE
        if X[0].shape[0] < self.BATCH_SIZE :

            predictions = convert_hot_to_squeeze_all(self.model(X).cpu())

        else :
            # Split the data into mini_batch to compute the prediction
            current_batch = 0
            # Number of images
            n = X[0].shape[0]
            # Indexes to map the images
            idxs = np.arange(n)
            # Predictions to return
            predictions = np.zeros((n, X[0].shape[2], X[0].shape[3]))
            self.model.eval()
            # Loop over the number of mini-batches
            while current_batch < n  :
                # Get the current indexes
                index = idxs[current_batch : min(n, current_batch+self.BATCH_SIZE)]
                # Get the current X
                current_x = [X[0][index,:].to(self.device) , X[1][index,:].to(self.device)]
                # Without gradient
                with torch.no_grad() :
                    # Compute the prediction
                    prediction = self.model(current_x).reshape(-1,self.N+1, 240,320)
                    # Increase the current batch
                    current_batch+= self.BATCH_SIZE
                    # Take the argmax for the prediction
                    prediction = convert_hot_to_squeeze_all(prediction.cpu())
                    # Add it to the output
                    predictions[index, :] = prediction

            self.model.train()
        return np.array(predictions)


    def score(self, X, y) :
        ''' Return the score for the inputs X and the output y '''
        prediction = self.predict(X)
        scores = self.evaluation.get_scores(prediction,y.cpu())
        return scores



    def train(self, n_epochs, to_load = False, to_save = True, epoch_start = 0 ) :
        ''' Train the models with the number of epochs '''

        if to_load :
            self.model.load_checkpoint()

        n = self.X_train[0].shape[0]
        # To store the loss
        writer = SummaryWriter(self.chkpt_dir)
        for e in range(epoch_start, n_epochs) :
            current_loss = 0

            idxs = np.arange(n)
            # Shuffle for the mini-batch
            np.random.shuffle(idxs)
            # Initialise the counter of the mini_batch
            current_batch = 0
            while current_batch < n :
                # Get the mini-batches
                batch_x , batch_y = get_batch(self.X_train, self.y_train, idxs ,\
                                              current_batch, self.BATCH_SIZE)
                # Convert the labels into one hot vectors
                batch_x = [batch_x[0].to(self.device),batch_x[1].to(self.device)]
                # batch_y = torch.from_numpy(one_hot_encode(batch_y, N= self.N+1)).to(self.device)
                batch_y = batch_y.to(self.device)
                # Increase the current batch
                current_batch+=self.BATCH_SIZE
                # Zero the parameters of the gradients
                self.optimizer.zero_grad()
                # Forward for the prediction
                batch_pred = self.model(batch_x)
                # Reshape the output
                batch_pred = batch_pred.reshape(-1, self.N+1, 240,320)
                # Reshape the target
                # batch_y = batch_y.reshape(-1, 240,320,  self.N+1).to(dtype=torch.long)
                # Compute the loss
                loss = self.criterion(batch_pred, torch.squeeze(batch_y).long())
                # Add it to store
                current_loss+=loss.detach().item()
                # Backward step
                loss.backward()
                self.optimizer.step()

            current_loss/=self.BATCH_SIZE
            self.loss_train[e] = current_loss
            writer.add_scalar("loss/", current_loss, e )

            if e % self.print_iter == 0 :
                scores = self.score(self.X_train , self.y_train)
                self.scores_train[e] = scores.copy()

                print('Epoch : {}  Loss : {:.2f}  Pixel acc : {:.2f} Mean acc : {:.2f} Mean IoU : {:.2f}'.format(e, current_loss,\
                                                                                                 scores['Pixel_accuracy'],scores['Mean_accuracy'],\
                                                                                                   scores['Mean_iou']))
                if self.X_val is not None :
                  scores_val = self.score(self.X_val , self.y_val)
                  print('Validation set : Pixel acc : {:.2f} Mean accuracy : {:.2f} Mean IoU : {}'.format(e,\
                                                                                                 scores_val['Pixel_accuracy'],\
                                                                                                 scores_val['Mean_accuracy'],\
                                                                                                   scores_val['Mean_iou']))

                  writer.add_scalar("scores/pixel_acc_val", scores_val['Pixel_accuracy'], e )
                  writer.add_scalar("scores/mean_acc_val", scores_val['Mean_accuracy'], e )
                  writer.add_scalar("scores/mean_iou_val", scores_val['Mean_iou'], e )

                  self.scores_val[e] = scores_val.copy()

                writer.add_scalar("scores/pixel_acc", scores['Pixel_accuracy'], e )
                writer.add_scalar("scores/mean_acc", scores['Mean_accuracy'], e )
                writer.add_scalar("scores/mean_iou", scores['Mean_iou'], e )




                self.model.save_checkpoint()

        writer.flush()
        writer.close()
