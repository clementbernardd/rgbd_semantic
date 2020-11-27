from importations import *


class Evaluation(object) :
    ''' Class that does the evaluation within 3 scores : Pixel accuracy, Mean accuracy and mean IoU '''
    def __init__(self) :
        ''' -score : A dictionnary with all the scores '''
        self.score = {}



    def get_scores(self, labels, prediction) :
        ''' Return a dictionnary with the 3 scores : Pixel accuracy, Mean accuracy and mean IoU '''
        pxl_acc = self.pixel_acc(labels, prediction)
        mean_acc = self.mean_acc(labels, prediction)
        mean_iou = self.mean_iou(labels, prediction)

        self.score['Pixel_accuracy'] = pxl_acc
        self.score['Mean_accuracy'] = mean_acc
        self.score['Mean_iou'] = mean_iou

        return self.score



    def pixel_acc(self, labels, prediction) :
        ''' Compute the pixel accuracy between the labels and the predictions '''
        # Convert the input into numpy
        if type(labels) == torch.Tensor :
            labels = labels.detach().numpy()
        if type(prediction) == torch.Tensor :
            prediction = prediction.detach().numpy()
        sim = (labels == prediction)
        acc = sim.sum() / sim.size

        return acc


    def mean_acc(self, labels,prediction, N = 22 ) :
        ''' Compute the mean accuracy between the labels and the predictions '''
        # Convert the input into numpy
        if type(labels) == torch.Tensor :
            labels = labels.detach().numpy()
        if type(prediction) == torch.Tensor :
            prediction = prediction.detach().numpy()

        accuracy = {}
        # Compute the accuracy for each class
        for classe in range(N) :

            # Get the logical array
            logical_target = np.zeros((labels.shape))
            logical_pred = 2 * np.ones((prediction.shape))

            # Index where the target is the given class
            target_idx = np.where(labels == classe)
            # Index where the prediction is the given class
            pred_idx = np.where(prediction == classe)
            # Complete the logical matrices
            logical_target[target_idx] = 1
            logical_pred[pred_idx] = 1

            # Compute the accuracy
            if np.sum(logical_target) != 0 :
                acc = len(np.where(logical_target == logical_pred)[0])/ np.sum(logical_target)
            else :
                acc = 0
            # Add it to the dictionnary
            accuracy[classe] = acc

        return np.array(list(accuracy.values())).mean()

    def mean_iou(self, labels, prediction, N = 22) :
        ''' Compute the Mean IoU between the labels and the predictions '''
        # Convert the input into numpy
        if type(labels) == torch.Tensor :
            labels = labels.detach().numpy()
        if type(prediction) == torch.Tensor :
            prediction = prediction.detach().numpy()

        iou_score = {}
        # Compute the IoU for each class
        for classe in range(N) :
            # Get the logical array
            logical_target = np.zeros((labels.shape))
            logical_pred = np.zeros((prediction.shape))

            # Index where the target is the given class
            target_idx = np.where(labels == classe)
            # Index where the prediction is the given class
            pred_idx = np.where(prediction == classe)
            # Complete the logical matrices
            logical_target[target_idx] = 1
            logical_pred[pred_idx] = 1
            # Get the intersection
            intersection = np.logical_and(logical_target, logical_pred)
            # Get the union
            union = np.logical_or(logical_target, logical_pred)
            if np.sum(union) == 0 :
                iou = 0
            else :
                # Compute the IoU score
                iou = np.sum(intersection) / np.sum(union)
            # Add it to the dictionnary
            iou_score[classe] = iou

        return np.array(list(iou_score.values())).mean()
