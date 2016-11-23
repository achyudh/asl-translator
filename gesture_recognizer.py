import os, gzip, pickle, random
from hand_localizer import get_hand
from gesture_classifier import predict_gesture_classifier
from skimage import io

class GestureRecognizer(object):

    """class to perform gesture recognition"""

    def __init__(self, data_directory):

        """
            data_directory : path like /home/sanket/mlproj/dataset/
            includes the dataset folder with '/'

            Initialize all your variables here
        """
        self.dataset_folder = data_directory
        # users_folder = os.listdir(dataset_folder)
        # users_folder = [name for name in users_folder if name[:4] == 'user' and 'pkl' not in name]
        self.window_size = 120
        self.models_folder = data_directory
        self.hand_clf = None
        self.gesture_clf = None


    def train(self, train_list):

        """
            train_list : list of users to use for training
            eg ["user_1", "user_2", "user_3"]

            The train function should train all your classifiers,
            both binary and multiclass on the given list of users
        """


    def recognize_gesture(self, image):

        """
            image : a 320x240 pixel RGB image in the form of a numpy array

            This function should locate the hand and classify the gesture.

            returns : (position, labels)

            position : a tuple of (x1,y1,x2,y2) coordinates of bounding box
                       x1,y1 is top left corner, x2,y2 is bottom right

            labels : a list of top 5 character predictions
                        eg - ['A', 'B', 'C', 'D', 'E']
        """
        result_arr = get_hand((image, self.hand_clf))
        ulx, uly, lrx, lry = result_arr
        predicted_gesture = predict_gesture_classifier(self.gesture_clf, [image[uly:lry, ulx:lrx]])

        return result_arr, predicted_gesture

    def save(**params):

        """
            save your GestureRecognizer to disk.
        """
        return


    def load_model(**params):

        """
            load your trained GestureRecognizer from disk with provided params
            Read - http://stackoverflow.com/questions/36901/what-does-double-star-and-star-do-for-parameters
        """
        pfile = gzip.open(params['model_directory'] + 'rfc_part2.pkl', 'r')
        hand_clf = pickle.load(pfile)
        pfile.close()
        pfile = gzip.open(params['model_directory'] + 'voting_part1.pkl', 'r')
        gesture_clf = pickle.load(pfile)
        pfile.close()
        gr = GestureRecognizer(params['data_directory'])
        gr.gesture_clf = gesture_clf
        gr.hand_clf = hand_clf
        return gr

# gr = GestureRecognizer.load_model(data_directory='Lite', model_directory='./')
# print(gr.recognize_gesture(io.imread('B9.jpg')))