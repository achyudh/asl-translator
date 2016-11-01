'''
Follow this directory structure for smooth functioning:

.
|--detect_hand.py
|--trial_data
	|--cropped
		|--1
			|--class specific images
		|--2
			|--class specific images
		|--3
			|--class specific images
		|--4
			|--class specific images
		|--5
			|--class specific images
	|--bounding_boxes.csv
	|--raw
		|--1
			|--class specific images
		|--2
			|--class specific images
		|--3
			|--class specific images
		|--4
			|--class specific images
		|--5
			|--class specific images
|--false
	|-- images extracted using the initialze() method in this code

'''
import os
from skimage import data,color,exposure
from skimage.feature import hog
from skimage.io import imread,imshow,imsave
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn import svm
from sklearn.svm import SVC
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from time import time
from sklearn.metrics import accuracy_score
from skimage.transform import resize
#Important file/folder paths
bounding_box_csv_file = '/trial_data/bounding_boxes.csv'
image_class_folder = '/trial_data/raw/'
image_class_folder_cropped = '/trial_data/cropped/'
current_folder = os.getcwd()

#HOG generator function
def hog_gen(path,image,display_img,show_size):
	if(path != 0 and image == 0):
		image = imread(path)
	#if(show_size):
		#print "target image shape is " + str(image.shape)
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

#Checks overlap of two rectangles 
def overlap(lx1,ly1,rx1,ry1,lx2,ly2,rx2,ry2):
	if(lx1 - rx2 > -40 or lx2 - rx1 > -40):
		return False
	if(ly1 - ry2 > -40 or ry1 - ly2 < 40):
		return False
	return True

#Extract not hand images from all images
def extract_false():
	df = pd.read_csv(current_folder+bounding_box_csv_file)
	classes = os.listdir(current_folder + image_class_folder)
	for _class in classes:
		dr = current_folder + image_class_folder+_class
		imgs = os.listdir(dr)

		for img in imgs:
			image = imread(dr+'/'+img)
			#print image.shape;
			lx1,ly1 = random.randint(0,1000000) %(320-128),random.randint(0,1000000) % (240 - 128)
			rx1,ry1 = lx1+128,ly1+128 

			ndf = df.loc[df['file_name'] == img]
			#print ndf
			for i,row in ndf.iterrows():
				lx2,ly2 = row['top_left_x'],row['top_left_y']
				rx2,ry2 = row['bottom_right_x'],row['bottom_right_y']
				counter = 0
				while(overlap(lx1,ly1,rx1,ry1,lx2,ly2,rx2,ry2)):
					counter +=1
					lx1,ly1 = random.randint(0,1000000) %(320-128),random.randint(0,1000000) % (240 - 128)
					rx1,ry1 = lx1+128,ly1+128 
					if(counter > 100):
						#print 'cant find suitable overlap '
						ly1,ry1,lx1,rx1 = ly1-ly1,ry1-ly1,lx1-lx1,rx1-lx1
						break
				cropped = image[ly1:ry1,lx1:rx1]
				imsave(current_folder+
					'/false/''nhand'+img,cropped)

#Load images of hand
def load_hand_images():
	l = []
	classes = os.listdir(current_folder + image_class_folder_cropped)
	for _class in classes:
		dr = current_folder + image_class_folder_cropped+_class
		imgs = os.listdir(dr)
		for img in imgs:
			l.append(hog_gen(0,imread(dr+'/'+img),False,False))
	return l

#Load images of not hand 
def load_nonhand_images():
	l = []
	imgs = os.listdir(current_folder + '/false/')
	for img in imgs:
		l.append(hog_gen(0,imread(current_folder + '/false/' + img),False,False))
	return l

#Generates images and saves them as numpy arrays
def initialize():
	extract_false()
	hand = load_hand_images()
	nhand = load_nonhand_images() 
	samples = np.vstack([hand,nhand])

	target_hand = [1]*len(hand)
	target_nhand = [0]*len(nhand)
	target_hand.extend(target_nhand)
	target = np.asarray(target_hand)
	np.save('samples',samples)
	np.save('targetvalue',target)

initialize()
X = np.load('samples.npy')
Y = np.load('targetvalue.npy')
X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25)

#PCA 
n_components = 300 #Number of features
pca = PCA(n_components = n_components,whiten = True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

#Fitting classfier using grid search
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

#Prediction on test data set
print("Predicting")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))
print(accuracy_score(y_test,y_pred))

#Checking on some random image
a = imread('4.jpg')
#Window coordinates
ulx = 0
uly = 0
lrx = 128
lry = 128
print(a.shape)
while(True):
	window = a[uly:lry,ulx:lrx]
	window = hog_gen(0,window,False,False)
	window = window.reshape(1,window.shape[0]*window.shape[1])
	w_pca  = pca.transform(window)
	pred = clf.predict(w_pca)
	print(pred)
	if(pred[0] == 1):
		imshow(a[uly:lry,ulx:lrx])
		plt.show()
	ulx +=20
	lrx +=20
	if(lrx> 320):
		ulx = 0
		lrx =  128
		lry += 10
		uly +=10
	if(lry > 240):
		break
	del window