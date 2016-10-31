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


def pickle_hog_arrays(outfilename, foldername, num_classes):
    pfile = gzip.open(outfilename, 'wb')
    hog_dataset = dict()
    for i0 in range(1, num_classes + 1):
        filelist = listdir(foldername + str(i0))
        for filename in filelist:
            hog_image = generate_hog_image(foldername + str(i0) + "/" + filename)
            hog_dataset[filename] = hog_image
    pickle.dump(hog_dataset, pfile)
    pfile.close()


def unpickle_hog_arrays(filename):
    pfile = gzip.open(filename, 'r')
    dataset = pickle.load(pfile)
    pfile.close()
    return dataset
