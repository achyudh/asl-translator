import pickle, gzip
from os import listdir
from util.preprocessing import *

def pickle_crossval_dataset(outfilename, x_train, y_train, x_test, y_test):
    pfile = gzip.open(outfilename, 'wb')
    dataset = [x_train, y_train, x_test, y_test]
    pickle.dump(dataset, pfile)
    pfile.close()


def unpickle_crossval_dataset(filename):
    pfile = gzip.open(filename, 'r')
    dataset = pickle.load(filename)
    return dataset[0], dataset[1], dataset[2], dataset[3]


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


def unpickle_features(filename):
    pfile = gzip.open(filename, 'r')
    dataset = pickle.load(pfile)
    pfile.close()
    return dataset


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


