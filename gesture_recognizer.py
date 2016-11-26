import os, gzip, pickle, random, gc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from skimage import color
from skimage.io import imread
from skimage.feature import hog
from skimage.transform import resize, rescale


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
        self.hand_clf, self.gesture_clf = train_hand_classifier(train_list, self.dataset_folder)

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
        result_arr = get_hand_coords((image, self.hand_clf))
        ulx, uly, lrx, lry = result_arr
        predicted_gesture = predict_gesture_classifier(self.gesture_clf, [image[uly:lry, ulx:lrx]])

        return result_arr, predicted_gesture

    def save(self,**params):

        """
            save your GestureRecognizer to disk.
        """
        pfile = gzip.open(params['model_directory']+'rfc_part2.pkl', 'wb')
        pickle.dump(self.hand_clf, pfile)
        pfile.close()
        pfile = gzip.open(params['model_directory']+'voting_part1.pkl', 'wb')
        pickle.dump(self.gesture_clf, pfile)
        pfile.close()
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


def percentage_overlap(lx1,ly1,rx1,ry1,lx2,ly2,rx2,ry2):
    x_overlap = max(0,min([rx1,rx2]) - max([lx1,lx2]) + 1)
    y_overlap = max(0,min([ry1,ry2]) - max([ly1,ly2]) + 1)
    overlap_area = x_overlap*y_overlap
    box_1_area = (rx1-lx1+1)*(ry1-ly1+1)
    box_2_area = (rx2-lx2+1)*(ry2-ly2+1)
    return overlap_area/float(box_1_area + box_2_area - overlap_area)


def get_windows(img, window_size=120, step_size=10):
    ulx,uly,lrx,lry = 0,0,window_size,window_size
    while(True):
        yield [img[uly:lry,ulx:lrx],ulx,uly,window_size]
        ulx+=step_size
        lrx+=step_size
        if(lrx> img.shape[1]):
            ulx,lrx = 0,window_size
            lry+=step_size
            uly+=step_size
        if(lry > img.shape[0]):
            break


def gen_imagescale(image, start_size=90, end_size=140, step_size=10, division_factor=120.0):
    for i in range(start_size,end_size,step_size):
        scale = i/division_factor
        yield rescale(image,(scale,scale)), scale


def train_hand_classifier(users_list, dataset_folder):
    training_data = []
    hand_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    gesture_clf = train_gesture_classifier(users_list, foldername=dataset_folder)
    for user in users_list:
        print("Generating HOGs for", user)
        df = pd.read_csv(dataset_folder + user + '/' + user + '_loc.csv')
        for f in os.listdir(dataset_folder + user):
            not_hand_crop = []
            hand_crop = []
            flag = 0
            if f.endswith(".jpg"):
                ndf = df.loc[df['image'] == user + '/' + f]
                lx2, ly2 = ndf['top_left_x'].item(), ndf['top_left_y'].item()
                rx2, ry2 = ndf['bottom_right_x'].item(), ndf['bottom_right_y'].item()
                input_img = imread(dataset_folder + user + '/' + f)
                hand_crop.append((input_img, (lx2, ly2, rx2, ry2)))
                for x_coord in range(20):
                    for y_coord in range(12):
                        ly1, ry1, lx1, rx1 = 10 * y_coord, 120 + 10 * y_coord, 10 * x_coord, 120 + 10 * x_coord
                        if percentage_overlap(lx1, ly1, rx1, ry1, lx2, ly2, rx2, ry2) >= 0.8:
                            hand_crop.append((input_img, (lx1, ly1, rx1, ry1)))
                        elif percentage_overlap(lx1, ly1, rx1, ry1, lx2, ly2, rx2, ry2) < 0.5:
                            flag += 1
                            if flag % 8 == 0:
                                not_hand_crop.append((input_img, (lx1, ly1, rx1, ry1)))
                hand_train = list(map(hog_gen_windows, hand_crop))
                not_hand_train = list(map(hog_gen_windows, not_hand_crop))
                del hand_crop
                del not_hand_crop
                hand_train = [(x, 1) for x in hand_train]
                not_hand_train = [(x, 0) for x in not_hand_train]
                training_data.extend(hand_train)
                training_data.extend(not_hand_train)

    random.shuffle(training_data)
    X_train = [x[0] for x in training_data]
    y_train = [x[1] for x in training_data]
    hand_clf.fit(X_train, y_train)
    # pfile = gzip.open('rfc_part2.pkl', 'wb')
    # pickle.dump(hand_clf, pfile)
    # pfile.close()
    return hand_clf, gesture_clf


def non_max_supression_fast(boxes,overlapThresh):
    if len(boxes) == 0:
        return np.array([[95, 55, 205, 165, 0]])
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = list()
    ulx,uly,lrx,lry = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    area = (lrx - ulx + 1)*(lry - uly + 1)
    prob = boxes[:,4]
    idxs = np.argsort(prob)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(ulx[i], ulx[idxs[:last]])
        yy1 = np.maximum(uly[i], uly[idxs[:last]])
        xx2 = np.minimum(lrx[i], lrx[idxs[:last]])
        yy2 = np.minimum(lry[i], lry[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("float")

def get_hand_coords(work_tuple):
    img, clf = work_tuple
    secondnms_list = []
    for scaled_img,scale in gen_imagescale(img):
        bounding_list = []


        win_list = [x for x in get_windows(scaled_img, window_size=120, step_size=10)]
        hog_win_list = [hog_gen(x[0]) for x in win_list]
        probs_list = clf.predict_proba(hog_win_list)
        for ip in range(len(probs_list)):
            if probs_list[ip][1] > 0.5:
                win = win_list[ip]
                bounding_list.append([win[1], win[2],win[1]+120, win[2]+120, probs_list[ip][1]])

        if len(bounding_list) > 0:
            boxes = np.vstack(bounding_list)
            final_box = non_max_supression_fast(boxes,0.3)
            for i in range(0,len(final_box)):
                mulx = final_box[i][0] / scale
                muly = final_box[i][1] / scale
                mlrx = mulx + (120/scale)
                mlry = muly + (120/scale)
                secondnms_list.append([mulx,muly,mlrx,mlry,final_box[i][4]])
    if len(secondnms_list) == 0:
        boxes = np.vstack([[130, 47, 250, 168, 0]])
    else:
        boxes = np.vstack(secondnms_list)
    final_box = non_max_supression_fast(boxes,0.3)

    return int(final_box[0][0]),int(final_box[0][1]),int(final_box[0][2]),int(final_box[0][3])


def generate_hog_features(image_arr):
    fd = hog(image_arr, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualise=False)
    return fd


def hog_gen_windows(work_tuple):
    image_arr, coords = work_tuple
    lx1,ly1,rx1,ry1 = coords
    if image_arr.ndim > 2:
        image_arr = resize(color.rgb2gray(image_arr)[ly1:ry1, lx1:rx1], (120, 120))
    hog_image_rescaled = generate_hog_features(image_arr)
    return hog_image_rescaled


def generate_training_set(work_tuple):
    img_arr = work_tuple[0]
    crop_before_x = work_tuple[1]
    crop_before_y = work_tuple[2]
    crop_after_x = work_tuple[3]
    crop_after_y = work_tuple[4]
    class_val = work_tuple[5]
    img_cropped = img_arr[crop_before_y:crop_after_y, crop_before_x:crop_after_x]
    img_transformed = resize(img_cropped, output_shape=(120, 120))
    hog_image = generate_hog_features(img_transformed)
    return hog_image, class_val


def train_gesture_classifier(userlist, foldername):
    """

    :param userlist:
    :param foldername:
    :return:
    """
    work_arr = list()
    class_alpha_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
                        'M': 11, 'N': 12, 'O': 13, 'P': 14, 'Q': 15, 'R': 16, 'S': 17, 'T': 18, 'U': 19, 'V': 20,
                        'W': 21, 'X': 22, 'Y': 23}

    print("Generating training features for gesture classifier...")
    for i0 in userlist:
        current_folder = foldername + i0 + '/'
        crop_df = pd.read_csv(current_folder + i0 + '_loc.csv', index_col=0, header=0)
        filelist = [x for x in os.listdir(current_folder) if x.endswith('.jpg')]
        for filename in filelist:
            img_arr = imread(current_folder + filename, as_grey=True)
            crop_before_x = crop_df.loc[i0 + '/' + filename, 'top_left_x']
            crop_before_y = crop_df.loc[i0 + '/' + filename, 'top_left_y']
            crop_after_x = crop_df.loc[i0 + '/' + filename, 'bottom_right_x']
            crop_after_y = crop_df.loc[i0 + '/' + filename, 'bottom_right_y']
            work_arr.append((img_arr, crop_before_x, crop_before_y, crop_after_x, crop_after_y, class_alpha_dict[filename[0]]))

    x_train = list(map(generate_training_set, work_arr))
    del work_arr
    print("Garbage collector deleted objects:", gc.collect())

    random.shuffle(x_train)
    y_train = [x[1] for x in x_train]
    x_train = [x[0] for x in x_train]
    print("Size of gesture classifier training set:", len(y_train))

    rfc_classifier = RandomForestClassifier(n_estimators=500, max_features='sqrt', n_jobs=8, warm_start=False)
    svc_classifier = SVC(cache_size=6000, kernel='linear', tol=1e-3, decision_function_shape='ovr', C=1, probability=True)
    voting_classifier = VotingClassifier(estimators=[('sv', svc_classifier), ('rf1', rfc_classifier)], voting='soft')
    voting_classifier.fit(x_train, y_train)
    print("Gesture classifier training complete.")

    return voting_classifier

def hog_gen(image, path=0):
    if path != 0 and image == 0:
        image = imread(path)
    if image.ndim > 2:
        image = color.rgb2gray(image)
    hog_image_rescaled = generate_hog_features(image)
    return hog_image_rescaled


def predict_gesture_classifier(voting_classifier, img_to_predict):
    work_arr = list()
    class_number_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
                         9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
                         17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}
    for color_img in img_to_predict:
        work_arr.append(color_img)
    x_test = list(map(generate_test_set, work_arr))
    y_probs = voting_classifier.predict_proba(x_test)
    y_best = np.argsort(y_probs, axis=1)[:,-5:]
    y_pred = np.empty(y_best.shape, dtype='str')
    for i0 in range(len(y_best)):
        for j0 in range(len(y_best[i0])):
            y_pred[i0,j0] = class_number_dict[y_best[i0,j0]]
    return y_pred

def generate_test_set(color_img):
    img_arr = color.rgb2gray(color_img)
    img_transformed = resize(img_arr, output_shape=(120, 120))
    hog_image = generate_hog_features(img_transformed)
    return hog_image

gr = GestureRecognizer('data/')
gr.train(['user_3','user_4'])
print(gr.recognize_gesture(imread('B9.jpg')))


