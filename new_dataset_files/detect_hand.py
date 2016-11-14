'''
Follow this directory structure for smooth functioning:

.
|--detect_hand.py
|--dataset
	|--user_<user_number>
		|--A0.jpg
		|--user_<user_number>_loc.csv 
		.
		.
	.
	.
	.

'''

from __future__ import print_function
import os
import pandas as pd
from skimage import color,exposure
from skimage.feature import hog
from skimage.io import imread,imshow,imsave
import matplotlib.pyplot as plt
from skimage.transform import resize,pyramid_gaussian
import multiprocessing as mp
import random
import numpy as np
from sklearn.cross_validation import train_test_split
from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

current_folder = os.getcwd()
dataset_folder = current_folder + '/dataset/'
users_folder = os.listdir(dataset_folder)

window_size = 110


def percentage_overlap(lx1,ly1,rx1,ry1,lx2,ly2,rx2,ry2):
	x_overlap = max(0,min([rx1,rx2]) - max([lx1,lx2]))
	y_overlap = max(0,min([ry1,ry2]) - max([ly1,ly2]))
	return (x_overlap * y_overlap )/ ((110.0 * 110.0) + ((rx1 - lx1)*(ry1 - ly1)) - (x_overlap * y_overlap ))
    

def makenpy(users_folder,dataset_folder):
	hands = []
	not_hand = []
	for users in users_folder:
		df = pd.read_csv(dataset_folder + users + '/' + users + '_loc.csv')
		for f in os.listdir(dataset_folder + users):
			if(f.endswith(".jpg")):
				ndf = df.loc[df['image'] == users+'/'+f]
				img = imread(dataset_folder + users + '/' + f)
				img = color.rgb2gray(img)
				hand_crop = img[ndf['top_left_y']:ndf['bottom_right_y'],ndf['top_left_x'] :ndf['bottom_right_x'] ]
				hands.append(resize(hand_crop,(110,110)))
				lx2,ly2 = random.randint(0,320-110),random.randint(0,240-110)
				rx2,ry2 = lx2+110,ly2+110
				for i,row in ndf.iterrows():
					lx1,ly1 = row['top_left_x'],row['top_left_y']
					rx1,ry1 = row['bottom_right_x'],row['bottom_right_y']

				while(percentage_overlap(lx1,ly1,rx1,ry1,lx2,ly2,rx2,ry2) > 0.4):
					lx2,ly2 = random.randint(0,320-110),random.randint(0,240-110)
					rx2,ry2 = lx2+110,ly2+110
				false_h = img[ly2:ry2,lx2:rx2]
				not_hand.append(false_h)
			else:
				print('done')
				continue

	print(len(not_hand))
	print(len(hands))
   	targets = [1]*len(hands)
	targets.extend([0]*len(not_hand))
	print(len(targets))
	pool = mp.Pool(4)
	hands = pool.map(hog_gen,hands)
	not_hand = pool.map(hog_gen,not_hand)
	np.save('samples',np.vstack([hands,not_hand]))
	np.save('target',np.asarray(targets))
    

def hog_gen(image,display_img=False,path=0):
	if(path != 0 and image == 0):
		image = imread(path)
	if(image.ndim > 2):
		image = color.rgb2gray(image)

	fd,hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(2, 2), visualise=True)

	if(display_img):
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
		ax1.axis('off')
		ax1.imshow(image, cmap=plt.cm.gray)
		ax1.set_title('Input image')
		ax1.set_adjustable('box-forced')
		# Rescale histogram for better display
		hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
		ax2.axis('off')
		ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
		ax2.set_title('Histogram of Oriented Gradients')
		ax1.set_adjustable('box-forced')
		plt.show()

	return hog_image

def load_npy(samples,target):
    X = np.load(samples)
    Y = np.load(target)
    print('loaded images')
    return X,Y
    
def train_classifier(X_train,y_train):
	print("Fitting the classifier to the training set")
	t0 = time()
	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
	              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced',random_state=10), param_grid,n_jobs=4)
	clf = clf.fit(X_train, y_train)
	print("done in %0.3fs" % (time() - t0))
	print("Best estimator found by grid search:")
	print(clf.best_estimator_)
	return clf
    
def get_pca(X_train,X_test,n_components):
    pca = PCA(n_components = n_components,whiten = True).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(X_train_pca.shape)
    return X_train_pca,X_test_pca,pca

def non_max_supression_fast(boxes,overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    ulx = boxes[:,0]
    uly = boxes[:,1]
    lrx = boxes[:,2]
    lry = boxes[:,3]
    area = (lrx - ulx + 1)*(lry - uly + 1)
    idxs = np.argsort(lry)
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
    return boxes[pick].astype("int")

def testOnImage(image,clf,pca):
	#Checking on some random image
	a = imread(image)
	bounding_list = []
	scale_f = 0
	#Window coordinates
	for scale in pyramid_gaussian(a,downscale = 1.1): 
	    ulx = 0
	    uly = 0
	    lrx = 110
	    lry = 110
	    print(scale.shape)
	    if(scale.shape[0] < 110 or scale.shape[1] < 110):
	        break
	    fig,ax = plt.subplots(ncols = 1,nrows = 1, figsize = (8,6))
	    ax.imshow(a)
	    
	    while(True):                             
	    	window = scale[uly:lry,ulx:lrx]
	    	window = hog_gen(window)
	    	window = window.reshape(1,window.shape[0]*window.shape[1])
	    	w_pca  = pca.transform(window)
	    	pred = clf.predict(w_pca)
	    	print(pred)
	    	if(pred[0] == 1):
	    		rect = mpatches.Rectangle((ulx, uly),110*(1.1**scale_f), 110*(1.1**scale_f),fill=False, edgecolor='red', linewidth=2)
	    		ax.add_patch(rect)
	           	bounding_list.append([ulx,uly,lrx + (110*(1.1**scale_f) - 110),lry + (110*(1.1**scale_f) - 110)])
	    	ulx +=20
	    	lrx +=20
	    	if(lrx> scale.shape[1]):
	    		ulx = 0
	    		lrx =  110
	    		lry += 20
	    		uly +=20
	    	if(lry > scale.shape[0]):
	    		break
	    	del window
	    scale_f+=1
	    plt.show()
	    
	boxes = np.vstack(bounding_list)
	final_box = non_max_supression_fast(boxes,0.3)

	fig,ax = plt.subplots(ncols = 1,nrows = 1, figsize = (8,6))
	ax.imshow(a)
	for i in range(0,len(final_box)):
	    rect = mpatches.Rectangle((final_box[i][0],final_box[i][1]),110,110,fill=False, edgecolor='red', linewidth=2)
	    ax.add_patch(rect)
	plt.show()


#makenpy(users_folder,dataset_folder)
X,Y = load_npy('samples.npy','target.npy')
X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25)
n_components = 1000 #Number of features
X_train_pca,X_test_pca,pca = get_pca(X_train,X_test)
trainedclf = train_classifier(X_train_pca,y_train)
y_pred = trainedclf.predict(X_test_pca)
print(accuracy_score(y_test,y_pred))


testOnImage('F3.jpg',trainedclf,pca)





