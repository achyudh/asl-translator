from gesture_classifier import *
import os, gzip, pickle, random
import pandas as pd
from skimage import color, exposure
from skimage.io import imread,imshow,imsave
from skimage.transform import resize,pyramid_gaussian,rescale
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
import numpy as np
from sklearn import metrics
from multiprocessing import Pool,cpu_count
import gc


def hog_gen(image, path=0):
    if path != 0 and image == 0:
        image = io.imread(path)
    if image.ndim > 2:
        image = color.rgb2gray(image)
    hog_image_rescaled = generate_hog_features(image)
    return hog_image_rescaled


def hog_file_gen(path):
    image = io.imread(path)
    if image.ndim > 2:
        image = color.rgb2gray(image)
    hog_image_rescaled = generate_hog_features(image)
    return hog_image_rescaled


def hog_gen_windows(work_tuple):
    image_arr, coords = work_tuple
    lx1,ly1,rx1,ry1 = coords
    if image_arr.ndim > 2:
        image_arr = resize(color.rgb2gray(image_arr)[ly1:ry1, lx1:rx1], (120, 120))
    hog_image_rescaled = generate_hog_features(image_arr)
    return hog_image_rescaled


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
        yield rescale(image,(scale,scale)),scale


def non_max_supression_fast(boxes,overlapThresh):
    if len(boxes) == 0:
        return np.array([[95, 55, 205, 165, 0]])
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = list()
    # print(boxes)
    ulx,uly,lrx,lry = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    area = (lrx - ulx + 1)*(lry - uly + 1)
    prob = boxes[:,4]
    idxs = np.argsort(prob)
    # print(idxs)
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
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("float")


def percentage_overlap(lx1,ly1,rx1,ry1,lx2,ly2,rx2,ry2):
    x_overlap = max(0,min([rx1,rx2]) - max([lx1,lx2]) + 1)
    y_overlap = max(0,min([ry1,ry2]) - max([ly1,ly2]) + 1)
    overlap_area = x_overlap*y_overlap
    box_1_area = (rx1-lx1+1)*(ry1-ly1+1)
    box_2_area = (rx2-lx2+1)*(ry2-ly2+1)
    return overlap_area/float(box_1_area + box_2_area - overlap_area)


def trainWithCV(users_folder, dataset_folder, clf, num_users=16, train=True):
    score_list = list()
    k_fold = 4  # Assume 5-fold CV
    num_groups = num_users//k_fold  # Number of groups to split the data into
    user_group_map = dict()  # Random mapping of users to groups
    num_users_in_group = [0 for i in range(num_groups)]  # Store the number of users in a group to ensure uniform distribution
    for i in range(3, num_users+4):
        if i == 8:
            continue
        rand_int = random.randrange(0, num_groups)
        while num_users_in_group[rand_int] >= k_fold:
            rand_int = random.randrange(0, num_groups)
        user_group_map[i] = rand_int
        num_users_in_group[rand_int] += 1

    print("Loading images...")

    for i0 in range(0,num_groups):
        print("-------\nGroup " + str(i0) + "\n=======")
        user_list = list()
        for user, group in user_group_map.items():
            if group != i0:
                user_list.append('user_'+str(user))

        # gesture_clf = train_gesture_classifier(user_list,foldername="data/")

        training_data = []
        testing_data = []
        for user in users_folder:
            print("Generating HOGs for", user)
            df = pd.read_csv(dataset_folder + user + '/' + user +'_loc.csv')
            for f in os.listdir(dataset_folder + user):
                not_hand_crop = []
                hand_crop = []
                flag = 0
                if f.endswith(".jpg"):
                    ndf = df.loc[df['image'] == user+'/'+f]
                    lx2,ly2 = ndf['top_left_x'].item(), ndf['top_left_y'].item()
                    rx2,ry2 = ndf['bottom_right_x'].item(),ndf['bottom_right_y'].item()
                    input_img = imread(dataset_folder + user + '/' + f)
                    hand_crop.append((input_img, (lx2,ly2,rx2,ry2)))
                    for x_coord in range(20):
                        for y_coord in range(12):
                            ly1, ry1, lx1, rx1 = 10*y_coord, 120+10*y_coord, 10*x_coord, 120+10*x_coord
                            # fig,ax = plt.subplots(1,1)
                            # ax.imshow(input_img)
                            # rect = mpatches.Rectangle((lx1,ly1),110,110,color='red',fill=False)
                            # ax.add_patch(rect)
                            # plt.show()
                            if percentage_overlap(lx1,ly1,rx1,ry1,lx2,ly2,rx2,ry2) >= 0.8:
                                hand_crop.append((input_img, (lx1,ly1,rx1,ry1)))
                            elif percentage_overlap(lx1,ly1,rx1,ry1,lx2,ly2,rx2,ry2) < 0.5:
                                flag +=1
                                if flag%8 ==0:
                                    not_hand_crop.append((input_img, (lx1,ly1,rx1,ry1)))

                    thread_pool = Pool(cpu_count())
                    hand_train = thread_pool.map(hog_gen_windows, hand_crop)
                    thread_pool.close()

                    thread_pool = Pool(cpu_count())
                    not_hand_train = thread_pool.map(hog_gen_windows, not_hand_crop)
                    thread_pool.close()
                    del hand_crop
                    del not_hand_crop
                    hand_train = [(x, 1) for x in hand_train]
                    not_hand_train = [(x, 0) for x in not_hand_train]

                    training_data.extend(hand_train)
                    training_data.extend(not_hand_train)

        print("Garbage collector deleted objects",gc.collect())

        random.shuffle(training_data)
        # random.shuffle(testing_data)
        X_train = [x[0] for x in training_data]
        y_train = [x[1] for x in training_data]
        # X_test = [x[0] for x in testing_data]
        # y_test = [x[1] for x in testing_data]

        print("X_train shape:", len(X_train))
        # print("X_test shape:", len(X_test))
        if train:
            clf.fit(X_train,y_train)
            pfile = gzip.open('rfc_part2_500trees_8mod.pkl', 'wb')
            pickle.dump(clf, pfile)
            pfile.close()
            exit()
        thread_pool = Pool(cpu_count())
        work_arr = [(x, clf) for x in X_test]
        result_arr = thread_pool.map(get_hand, work_arr)
        thread_pool.close()
        # print(len(result_arr), len(y_test))
        score = hscore = 0
        for i2 in range(len(result_arr)):
            ulx, uly, lrx, lry = result_arr[i2]
            if percentage_overlap(ulx, uly, lrx, lry, y_test[i2][0], y_test[i2][1], y_test[i2][2], y_test[i2][3]) > 0.5:
                hscore += 1
                predicted_gesture = predict_gesture_classifier(gesture_clf, [X_test[i2][uly:lry, ulx:lrx]])
                if y_test[i2][4] in predicted_gesture[0]:
                    score += 1
        del result_arr
        del work_arr
        print("Garbage collector deleted objects:", gc.collect())
        print("Overall score:", score/len(testing_data))
        print("Localization score:", hscore/len(testing_data))
        score_list.append(score/len(testing_data))
    print("Average score:", sum(score_list)/len(score_list))


def get_hand(work_tuple):
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
            # print(boxes)
            final_box = non_max_supression_fast(boxes,0.3)
            for i in range(0,len(final_box)):
                # print(final_box[i])
                mulx = final_box[i][0] / scale
                muly = final_box[i][1] / scale
                mlrx = mulx + (120/scale)
                mlry = muly + (120/scale)
                secondnms_list.append([mulx,muly,mlrx,mlry,final_box[i][4]])
                #rect = mpatches.Rectangle((final_box[i][0],final_box[i][1]),final_box[i][2] -final_box[i][0] ,final_box[i][3]-final_box[i][1],fill=False, edgecolor='red', linewidth=2)
                #ax.add_patch(rect)
                #plt.show()
    # fig,ax = plt.subplots(ncols = 1,nrows = 1)
    # ax.imshow(img)
    # for i in range(0,len(secondnms_list)):
    #     rect = mpatches.Rectangle((int(secondnms_list[i][0]),int(secondnms_list[i][1])),int(secondnms_list[i][2] -secondnms_list[i][0]) ,int(secondnms_list[i][3]-secondnms_list[i][1]),fill=False, edgecolor='red', linewidth=2)
    #     ax.add_patch(rect)
    # plt.show()
    if len(secondnms_list) == 0:
        boxes = np.vstack([[130, 47, 250, 168, 0]])
    else:
        boxes = np.vstack(secondnms_list)
    final_box = non_max_supression_fast(boxes,0.3)
    # fig,ax = plt.subplots(ncols = 1,nrows = 1)
    # ax.imshow(img)
    # for i in range(0,len(final_box)):
    #     rect = mpatches.Rectangle((int(final_box[i][0]),int(final_box[i][1])),int(final_box[i][2] -final_box[i][0]) ,int(final_box[i][3]-final_box[i][1]),fill=False, edgecolor='red', linewidth=2)
    #     ax.add_patch(rect)
    # plt.show()
    return int(final_box[0][0]),int(final_box[0][1]),int(final_box[0][2]),int(final_box[0][3])


# print("Initializing...")
# current_folder = os.getcwd()
# dataset_folder = current_folder + '/data/'
# users_folder = os.listdir(dataset_folder)
# users_folder = [name for name in users_folder if name[:4] == 'user' and 'pkl' not in name]
# window_size = 120
# models_folder = current_folder
#
# # get_hand(imread("A9.jpg"),clf)
# clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
#
# trainWithCV(users_folder, dataset_folder, clf, num_users=16, train=True)
