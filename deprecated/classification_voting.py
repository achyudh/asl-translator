from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from util.dataset_io import class_alpha_dict
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
    hog_fd_dataset = unpickle_features("data/cropped_hog_fd.pkl")
    daisy_dataset = unpickle_features("data/cropped_daisy.pkl")
    for t0 in range(0, num_groups):
        x_train = list()
        x_crossval = list()
        for i0 in range(1, num_classes+1):
            filelist = listdir(foldername + str(i0))
            for filename in filelist:
                hog_image = hog_dataset[filename]
                hog_fd = hog_fd_dataset[filename]
                daisy_features = daisy_dataset[filename]
                if user_group_map[int(filename.split('_')[0])] == t0:
                    x_crossval.append((hog_image, hog_fd, daisy_features, i0-1))
                else:
                    x_train.append((hog_image, hog_fd, daisy_features, i0-1))
        random.shuffle(x_train)
        random.shuffle(x_crossval)
        y_train = [x[3] for x in x_train]
        x_train = [np.concatenate([x[0].ravel(), x[1].ravel()]) for x in x_train]
        y_crossval = [x[3] for x in x_crossval]
        x_crossval = [np.concatenate([x[0].ravel(), x[1].ravel()]) for x in x_crossval]

        print("Size of Training Set:", len(x_train), "\nSize of Crossval Set:", len(x_crossval))

        rfc_classifier1 = RandomForestClassifier(n_estimators=1000, max_features='sqrt', n_jobs=6, warm_start=False)
        svc_classifier = SVC(cache_size=6000, kernel='linear', tol=1e-3, decision_function_shape='ovr', C=1, probability=True)
        voting_classifier = VotingClassifier(estimators=[('sv', svc_classifier), ('rf1', rfc_classifier1)], voting='soft')
        voting_classifier.fit(x_train, y_train)
        # print("1:", voting_classifier.score(x_crossval, y_crossval))
        y_probs = voting_classifier.predict_proba(x_crossval)
        best_n = np.argsort(y_probs, axis=1)[:,-5:]
        true_class = 0
        for i2 in range(len(y_crossval)):
            if y_crossval[i2] in best_n[i2]:
                true_class += 1
        curr_score = true_class/len(y_crossval)
        print("5:", curr_score)
        voting_classifier_scores.append(curr_score)

    print(sum(voting_classifier_scores)/len(voting_classifier_scores))

