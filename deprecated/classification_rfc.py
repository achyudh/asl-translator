from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.cross_validation import cross_val_score
from os import listdir
import numpy as np
import gzip, pickle
import random


def unpickle_features(filename):
    pfile = gzip.open(filename, 'r')
    dataset = pickle.load(pfile)
    pfile.close()
    return dataset


def classify_images(foldername="data/cropped/", num_users=16, num_classes=24):
    """

    :param foldername:
    :param num_users:
    :param num_classes:
    :return:
    """
    k_fold = 4  # Assume 5-fold CV
    num_groups = num_users//k_fold  # Number of groups to split the data into
    voting_classifier_scores = list()
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
    print(num_users_in_group)
    print("Loading images...")
    hog_dataset = unpickle_features("data/cropped_hog_4x4.pkl")
    daisy_dataset = unpickle_features("data/cropped_daisy.pkl")
    lbp_dataset = unpickle_features("data/cropped_lbp.pkl")
    for t0 in range(0, num_groups):
        x_train = list()
        x_crossval = list()
        for i0 in range(1, num_classes+1):
            filelist = listdir(foldername + str(i0))
            for filename in filelist:
                hog_image = hog_dataset[filename]
                daisy_features = daisy_dataset[filename]
                lbp_features = lbp_dataset[filename]
                if user_group_map[int(filename.split('_')[0])] == t0:
                    x_crossval.append((hog_image, daisy_features, lbp_features, i0-1))
                else:
                    x_train.append((hog_image, daisy_features, lbp_features, i0-1))
        random.shuffle(x_train)
        random.shuffle(x_crossval)
        y_train = [x[3] for x in x_train]
        x_train = [np.concatenate([x[0].ravel(), x[1].ravel(), x[2].ravel()]) for x in x_train]
        y_crossval = [x[3] for x in x_crossval]
        x_crossval = [np.concatenate([x[0].ravel(), x[1].ravel(), x[2].ravel()]) for x in x_crossval]
        print("Size of Training Set:", len(x_train), "\nSize of Crossval Set:", len(x_crossval))

        # print("RFC1")
        rfc_classifier1 = RandomForestClassifier(n_estimators=3000, max_features='sqrt', n_jobs=7, warm_start=True)
        # rfc_classifier1.fit(x_train, y_train)
        # curr_score = rfc_classifier1.score(x_crossval, y_crossval)
        # print(curr_score)
        #
        # print("RFC2")
        rfc_classifier2 = RandomForestClassifier(n_estimators=5000, max_features='log2', n_jobs=7, warm_start=True)
        # rfc_classifier2.fit(x_train, y_train)
        # curr_score = rfc_classifier2.score(x_crossval, y_crossval)
        # print(curr_score)

        # print("RFC3")
        # rfc_classifier3 = RandomForestClassifier(n_estimators=500, max_features='sqrt', n_jobs=7, warm_start=True)
        # rfc_classifier3.fit(x_train, y_train)
        # curr_score = rfc_classifier3.score(x_crossval, y_crossval)
        # print(curr_score)

        print("VotingClassifier")
        voting_classifier = VotingClassifier(estimators=[('rfcsq1', rfc_classifier1), ('rfclog', rfc_classifier2), ('rfcsq2', rfc_classifier3)],
                                             voting='soft')
        voting_classifier.fit(x_train, y_train)
        curr_score = voting_classifier.score(x_crossval, y_crossval)
        print(curr_score)
        voting_classifier_scores.append(curr_score)

    print(sum(voting_classifier_scores)/len(voting_classifier_scores))

    print(sum(voting_classifier_scores)/len(voting_classifier_scores))

classify_images()
