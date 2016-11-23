from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from skimage import io, transform, exposure, color
from skimage.feature import hog
from multiprocessing import Pool
from os import listdir
import numpy as np
import pandas as pd
import random
import gc, pickle, gzip
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from skimage import io, transform, exposure, color
from skimage.feature import hog
from multiprocessing import Pool
from os import listdir
import numpy as np
import pandas as pd
import random
import gc, pickle, gzip

class_alpha_dict = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9, 'L':10, 'M':11, 'N':12,
                    'O':13, 'P':14, 'Q':15, 'R':16, 'S':17, 'T':18, 'U':19, 'V':20, 'W':21, 'X':22, 'Y': 23}

class_number_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'K', 10:'L', 11:'M', 12:'N',
                     13:'O', 14:'P', 15:'Q', 16:'R', 17:'S', 18:'T', 19:'U', 20:'V', 21:'W', 22:'X', 23:'Y'}


def generate_hog_features(image_arr):
    fd = hog(image_arr, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualise=False)
    # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    return fd


def generate_training_set(work_tuple):
    img_arr = work_tuple[0]
    crop_before_x = work_tuple[1]
    crop_before_y = work_tuple[2]
    crop_after_x = work_tuple[3]
    crop_after_y = work_tuple[4]
    class_val = work_tuple[5]
    img_cropped = img_arr[crop_before_y:crop_after_y, crop_before_x:crop_after_x]
    img_transformed = transform.resize(img_cropped, output_shape=(120, 120))
    hog_image = generate_hog_features(img_transformed)
    return hog_image, class_val


def generate_test_set(color_img):
    img_arr = color.rgb2gray(color_img)
    img_transformed = transform.resize(img_arr, output_shape=(120, 120))
    hog_image = generate_hog_features(img_transformed)
    return hog_image


def train_gesture_classifier(userlist, foldername="data/"):
    """

    :param userlist:
    :param foldername:
    :return:
    """
    work_arr = list()
    print("Generating training features for gesture classifier...")
    for i0 in userlist:
        current_folder = foldername + i0 + '/'
        crop_df = pd.read_csv(current_folder + i0 + '_loc.csv', index_col=0, header=0)
        filelist = [x for x in listdir(current_folder) if x.endswith('.jpg')]
        for filename in filelist:
            img_arr = io.imread(current_folder + filename, as_grey=True)
            crop_before_x = crop_df.loc[i0 + '/' + filename, 'top_left_x']
            crop_before_y = crop_df.loc[i0 + '/' + filename, 'top_left_y']
            crop_after_x = crop_df.loc[i0 + '/' + filename, 'bottom_right_x']
            crop_after_y = crop_df.loc[i0 + '/' + filename, 'bottom_right_y']
            work_arr.append((img_arr, crop_before_x, crop_before_y, crop_after_x, crop_after_y, class_alpha_dict[filename[0]]))

    thread_pool = Pool(8)
    x_train = thread_pool.map(generate_training_set, work_arr)
    thread_pool.close()
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


def predict_gesture_classifier(voting_classifier, img_to_predict):
    """

    :param voting_classifier:
    :param img_to_predict:
    :return:
    """
    # print("Predicting gestures...")
    work_arr = list()
    for color_img in img_to_predict:
        work_arr.append(color_img)
    #thread_pool = Pool(8)
    x_test = list(map(generate_test_set, work_arr))
    #thread_pool.close()
    y_probs = voting_classifier.predict_proba(x_test)
    y_best = np.argsort(y_probs, axis=1)[:,-5:]
    y_pred = np.empty(y_best.shape, dtype='str')
    for i0 in range(len(y_best)):
        for j0 in range(len(y_best[i0])):
            y_pred[i0,j0] = class_number_dict[y_best[i0,j0]]
    return y_pred

# scores = list()
# grouplist_int = [[3, 4, 5, 6], [7, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19]]
# grouplist = [['user_' + str(x) for x in y] for y in grouplist_int]

# userlist = np.concatenate([x for x in grouplist])
# vclf = train_gesture_classifier(userlist)
#
# pfile = gzip.open('voting_part1.pkl', 'wb')
# pickle.dump(vclf, pfile)
# pfile.close()
# for ic in range(4):
#     testlist = grouplist[ic]
#     userlist = np.concatenate([x for x in grouplist if x!=testlist])
#     vclf = train_gesture_classifier(userlist)
#     img_arr = list()
#     y_true = list()
#     for i0 in range(1, 24):
#         ctr = 0
#         filelist = listdir("data/cropped/" + str(i0))
#         for filename in filelist:
#             if int(filename.split('_')[0]) in grouplist_int[ic]:
#                 img_arr.append(io.imread("data/cropped/" + str(i0) + "/" + filename))
#                 y_true.append(class_number_dict[i0-1])
#
#     y_pred = predict_gesture_classifier(vclf, img_arr)
#
#     true_class = 0
#     for i2 in range(len(y_true)):
#         # print(y_true[i2], y_pred[i2], y_true[i2] in y_pred[i2])
#         if y_true[i2] in y_pred[i2]:
#             true_class += 1
#     curr_score = true_class / len(y_true)
#     scores.append(curr_score)
#     print(curr_score)
# print("Avg:", sum(scores)/len(scores))
