import pickle, gzip, skimage
from os import listdir, path, makedirs
from util.preprocessing import *
import pandas as pd
import numpy as np



def pickle_crossval_dataset(outfilename, x_train, y_train, x_test, y_test):
    pfile = gzip.open(outfilename, 'wb')
    dataset = [x_train, y_train, x_test, y_test]
    pickle.dump(dataset, pfile)
    pfile.close()


def load_npy(samples,target):
    X = np.load(samples)
    Y = np.load(target)
    print('Loaded images')
    return X,Y


def unpickle_crossval_dataset(filename):
    pfile = gzip.open(filename, 'r')
    dataset = pickle.load(pfile)
    return dataset[0], dataset[1], dataset[2], dataset[3]


def unpickle_features(filename):
    pfile = gzip.open(filename, 'r')
    dataset = pickle.load(pfile)
    pfile.close()
    return dataset


def pickle_hog_features(outfilename, foldername, num_classes):
    pfile = gzip.open(outfilename, 'wb')
    hog_dataset = dict()
    for i0 in range(1, num_classes + 1):
        filelist = listdir(foldername + str(i0))
        for filename in filelist:
            hog_image = generate_hog_features(foldername + str(i0) + "/" + filename)
            hog_dataset[filename] = hog_image
    pickle.dump(hog_dataset, pfile)
    pfile.close()


def pickle_daisy_features(outfilename, foldername, num_classes):
    pfile = gzip.open(outfilename, 'wb')
    daisy_dataset = dict()
    for i0 in range(1, num_classes + 1):
        filelist = listdir(foldername + str(i0))
        for filename in filelist:
            daisy_descs = generate_daisy_features(foldername + str(i0) + "/" + filename)
            daisy_dataset[filename] = daisy_descs
    pickle.dump(daisy_dataset, pfile)
    pfile.close()


def pickle_lbp_features(outfilename, foldername, num_classes):
    pfile = gzip.open(outfilename, 'wb')
    lbp_dataset = dict()
    for i0 in range(1, num_classes + 1):
        filelist = listdir(foldername + str(i0))
        for filename in filelist:
            lbp_features = generate_lbp_features(foldername + str(i0) + "/" + filename)
            lbp_dataset[filename] = lbp_features
    pickle.dump(lbp_dataset, pfile)
    pfile.close()


def crop_images(foldername, num_classes):
    for i0 in range(1, num_classes + 1):
        if not path.exists(foldername + 'cropped/' + str(i0)):
            makedirs(foldername + 'cropped/' + str(i0))
    user_folders = listdir(foldername)
    user_folders.remove('cropped')
    for user_folder in user_folders:
        curr_folder = foldername + user_folder + '/'
        crop_df = pd.read_csv(curr_folder + user_folder + '_loc.csv', index_col=0, header=0)
        filelist = listdir(curr_folder)
        for filename in filelist:
            file_class = str(class_alpha_dict[filename[0].upper()])
            user_num = user_folder.split('_')[1]
            if filename.split('.')[1] == 'csv':
                continue
            img_arr = io.imread(curr_folder+filename)
            crop_before_x = crop_df.loc[user_folder+'/'+filename, 'top_left_x']
            crop_before_y = crop_df.loc[user_folder+'/'+filename, 'top_left_y']
            crop_after_x = crop_df.loc[user_folder+'/'+filename, 'bottom_right_x']
            crop_after_y = crop_df.loc[user_folder+'/'+filename, 'bottom_right_y']
            img_cropped = img_arr[crop_before_y:crop_after_y, crop_before_x:crop_after_x]
            img_transformed = skimage.transform.resize(img_cropped, output_shape=(110, 110))
            skimage.io.imsave(foldername + 'cropped/' + file_class + "/" + user_num + '_' + filename, img_transformed)


def create_userpkl(users_folder,dataset_folder,data_dict):
    for users in users_folder:
        u_dict = {}
        for f in listdir(dataset_folder + users):
            if(f.endswith(".jpg")):
                u_dict[users +'/'+f] = data_dict[users +'/'+f]
        print("pickling " + users)
        f = gzip.open('data//'+users+'.pkl','wb',compresslevel=1)
        pickle.dump(u_dict,f)
        f.close()
