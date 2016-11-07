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


def classify_images(foldername="data/cropped/", num_users=20, num_classes=5):
    """

    :param foldername:
    :param num_users:
    :param num_classes:
    :return:
    """
    k_fold = 5  # Assume 5-fold CV
    num_groups = num_users//k_fold  # Number of groups to split the data into
    voting_classifier_scores = list()
    user_group_map = dict()  # Random mapping of users to groups
    num_users_in_group = [0 for i in range(num_groups)]  # Store the number of users in a group to ensure uniform distribution

    for i in range(0, num_users):
        rand_int = random.randrange(0, num_groups)
        while num_users_in_group[rand_int] >= k_fold:
            rand_int = random.randrange(0, num_groups)
        user_group_map[i] = rand_int
        num_users_in_group[rand_int] += 1

    print("Loading images...")
    # hog_dataset = unpickle_features("data/cropped_hog_4x4.pkl")
    # daisy_dataset = unpickle_features("data/cropped_daisy.pkl")
    # lbp_dataset = unpickle_features("data/cropped_lbp.pkl")
    # for t0 in range(0, num_groups):
    #     x_train = list()
    #     x_crossval = list()
    #     for i0 in range(1, num_classes+1):
    #         filelist = listdir(foldername + str(i0))
    #         for filename in filelist:
    #             hog_image = hog_dataset[filename]
    #             daisy_features = daisy_dataset[filename]
    #             lbp_features = lbp_dataset[filename]
    #             if user_group_map[int(filename.split('_')[0])-1] == t0:
    #                 x_crossval.append((hog_image, daisy_features, lbp_features, i0-1))
    #             else:
    #                 x_train.append((hog_image, daisy_features, lbp_features, i0-1))
    #     random.shuffle(x_train)
    #     random.shuffle(x_crossval)
    #     y_train = [x[3] for x in x_train]
    #     x_train = [np.concatenate([x[0].ravel(), x[1].ravel(), x[2].ravel()]) for x in x_train]
    #     y_crossval = [x[3] for x in x_crossval]
    #     x_crossval = [np.concatenate([x[0].ravel(), x[1].ravel(), x[2].ravel()]) for x in x_crossval]
    #     print("Size of Training Set:", len(x_train), "\nSize of Crossval Set:", len(x_crossval))
    #
    #     rfc_classifier = RandomForestClassifier(n_estimators=1000, max_features='sqrt', n_jobs=4)
    #     gradient_boosting_classifier = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0, max_depth=3)
    #     svc_classifier = SVC(cache_size=2000, kernel='linear', tol=1e-3, decision_function_shape='ovr', C=1, probability=True)
    #     voting_classifier = VotingClassifier(estimators=[('rf', rfc_classifier), ('gb', gradient_boosting_classifier), ('sv', svc_classifier)],
    #                                          voting='hard')
    #     voting_classifier.fit(x_train, y_train)
    #     curr_score = voting_classifier.score(x_crossval, y_crossval)
    #     print(curr_score)
    #     voting_classifier_scores.append(curr_score)

    hog_dataset = unpickle_features("data/cropped_hog_4x4.pkl")
    daisy_dataset = unpickle_features("data/cropped_daisy.pkl")
    lbp_dataset = unpickle_features("data/cropped_lbp.pkl")
    x_train = list()
    group_labels = list()
    for i0 in range(1, num_classes + 1):
            filelist = listdir(foldername + str(i0))
            for filename in filelist:
                hog_image = hog_dataset[filename]
                daisy_features = daisy_dataset[filename]
                lbp_features = lbp_dataset[filename]
                x_train.append((hog_image, daisy_features, lbp_features, i0-1))
                group_labels.append(int(filename.split('_')[0])-1)
    y_train = [x[3] for x in x_train]
    x_train = [np.concatenate([x[0].ravel(), x[1].ravel(), x[2].ravel()]) for x in x_train]

    rfc_classifier = RandomForestClassifier(n_estimators=1000, max_features='sqrt', n_jobs=4)
    gradient_boosting_classifier = GradientBoostingClassifier(n_estimators=500, learning_rate=1.0, max_depth=3)
    svc_classifier = SVC(cache_size=2000, kernel='linear', tol=1e-3, decision_function_shape='ovr', C=1, probability=True)
    voting_classifier = VotingClassifier(estimators=[('rf', rfc_classifier), ('sv', svc_classifier),
                                                     ('gb', gradient_boosting_classifier)], voting='hard')
    scores = cross_val_score(estimator=voting_classifier, X=x_train, y=y_train, cv=5, verbose=1, n_jobs=4)
    print(scores)

classify_images()
