from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from util.dataset_io import *
from os import listdir
import random


def classify_images(foldername="data/cropped/", num_users=20, num_classes=5):
    """

    :param foldername:
    :param num_users:
    :param num_classes:
    :return:
    """
    k_fold = 5  # Assume 5-fold CV
    num_groups = num_users//k_fold  # Number of groups to split the data into
    svc_scores = list()
    rfc_scores = list()
    user_group_map = dict()  # Random mapping of users to groups
    num_users_in_group = [0 for i in range(num_groups)]  # Store the number of users in a group to ensure uniform distribution

    for i in range(0, num_users):
        rand_int = random.randrange(0, num_groups)
        while num_users_in_group[rand_int] >= 5:
            rand_int = random.randrange(0, num_groups)
        user_group_map[i] = rand_int
        num_users_in_group[rand_int] = 1

    print("Loading images...")
    hog_dataset = unpickle_hog_arrays("data/cropped_hog.pkl")
    for t0 in range(0, num_groups):
        print("ITERATION:", t0+1, "\n----------")
        x_train = list()
        x_crossval = list()
        for i0 in range(1, num_classes+1):
            filelist = listdir(foldername + str(i0))
            for filename in filelist:
                hog_image = hog_dataset[filename]
                if user_group_map[int(filename.split('_')[0])-1] == t0:
                    x_crossval.append((hog_image, i0))
                else:
                    x_train.append((hog_image, i0))

        random.shuffle(x_train)
        random.shuffle(x_crossval)
        y_train = [x[1] for x in x_train]
        x_train = [x[0].ravel() for x in x_train]
        y_crossval = [x[1] for x in x_crossval]
        x_crossval = [x[0].ravel() for x in x_crossval]
        print("Size of Training Set:", len(x_train), "\nSize of Crossval Set:", len(x_crossval))

        # RFC Classifier
        print("RFC Classifier Accuracy:")
        rfc_classifier = RandomForestClassifier()
        rfc_classifier.fit(x_train, y_train)
        curr_score = rfc_classifier.score(x_crossval, y_crossval)
        print(curr_score)
        rfc_scores.append(curr_score)

        # SVC Classifier
        print("SVC Classifier Accuracy:")
        svc_classifier = SVC()
        svc_classifier.fit(x_train, y_train)
        curr_score = svc_classifier.score(x_crossval, y_crossval)
        print(curr_score)
        svc_scores.append(curr_score)
        print("\n")

    print("Averge RFC accuracy:", sum(rfc_scores)/num_groups)
    print("Averge SVC accuracy:", sum(svc_scores)/num_groups)

classify_images()
