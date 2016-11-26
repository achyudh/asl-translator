import sys
import json
import imp
import numpy as np
from skimage import io
from time import time
from functools import partial
from multiprocessing import Pool, cpu_count

class Evaluator(object):

	def __init__(self, data_directory):

		"""
			data_directory : path like /home/sanket/mlproj/dataset/
			includes the dataset folder with '/'
		"""

		self.data_directory = data_directory

	@staticmethod
	def IOU(box1, box2):

		"""
			returns IOU score for box1 and box2
		"""

		xmin_1, ymin_1, xmax_1, ymax_1 = map(int, box1)
		xmin_2, ymin_2, xmax_2, ymax_2 = map(int, box2)

		# Evaluator to floats : YOU SHALL NOT PASS!!!
		# Seriously, don't pass floats.

		"""
			box has format (x1,y1,x2,y2)
			where x1,y1 is top left corner
			and x2,y2 is bottom right corner
		"""

		dx = min(xmax_1, xmax_2) - max(xmin_1, xmin_2)
		dy = min(ymax_1, ymax_2) - max(ymin_1, ymin_2)
		
		if (dx >= 0) and (dy >= 0):
			intersection = dx * dy

		else:
			intersection = 0

		area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)
		area_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)

		union = area_1 + area_2 - intersection

		try:
			iou_score = float(intersection) / union
		except:
			iou_score = 0.0

		return iou_score

	def load_images(self, test_list):

		"""
			train_list : list of users to use for testing
			eg ["user_1", "user_2", "user_3"]
		"""

		self.image_list = []

		for user in test_list:

			csv = "%s%s/%s_loc.csv" % (self.data_directory, user, user)

			with open(csv) as fh:
				data = [line.strip().split(',') for line in fh]

			for line in data[1:]:
				
				img_path, x1,y1,x2,y2, = line
				pos = tuple(map(int,(x1,y1,x2,y2)))
				letter = img_path[-6]

				img = io.imread("%s%s" % (self.data_directory, img_path))

				self.image_list.append((img, pos, letter))

	def evaluate(self, gr, parallel = False):

		"""
			gr : trained instance of GestureRecognizer
		"""

		if parallel:
			return self.evaluate_parallel(gr)

		total_samples = len(self.image_list)

		correct_localizations = 0
		correct_classifications = 0

		for i, (img, pos, letter) in enumerate(self.image_list):

			print ("processing image %d of %d" % (i+1, total_samples))

			bbox, labels = gr.recognize_gesture(img)

			iou_score = Evaluator.IOU(pos, bbox)

			if iou_score >= 0.5:
				correct_localizations += 1

			if letter in labels and len(labels) <= 5:
				correct_classifications += 1

		correct_localizations /= float(total_samples)
		correct_classifications /= float(total_samples)
		
		loc_score = 6.5 * correct_localizations
		clf_score = 3.5 * correct_classifications

		return loc_score, clf_score

	def evaluate_parallel(self, gr):

		wrapper = partial(evaluate_img, gr = gr)

		pool = Pool(processes = int(sys.argv[1]))
		results = pool.map(wrapper, enumerate(self.image_list))
		pool.close()
		pool.join()

		results = np.array(results)
		correct_localizations = results[:,0].sum() / float(len(results))
		correct_classifications = results[:,1].sum() / float(len(results))

		loc_score = 6.5 * correct_localizations
		clf_score = 3.5 * correct_classifications

		return loc_score, clf_score

def evaluate_img(param1, gr):

	i, (img, pos, letter) = param1

	print ("processing image %d of 960" % (i+1))

	bbox, labels = gr.recognize_gesture(img)
	iou_score = Evaluator.IOU(pos, bbox)

	loc, clf = 0,0

	if iou_score >= 0.5:
		loc = 1
	
	if letter in labels and len(labels) <= 5:
		clf = 1

	return loc,clf


if __name__ == "__main__":

	"""
		An example of how the evaluator script will be used
	"""
	
	test_list = ["user_1", "user_2", "user_8", "user_20"]

	evaluator = Evaluator("/home/oversmart/mlproject/dataset/")
	evaluator.load_images(test_list)

	with open("params.json") as fh:
		params = json.loads(fh.read())

	from gesture_recognizer import GestureRecognizer
	
	gr = GestureRecognizer.load_model(**params)
	
	t = time()
	loc_score, clf_score = evaluator.evaluate(gr, parallel = True)
	
	print ("Evaluation done, took %f seconds" % (time() - t))
	print ("Localization Score : %f" % loc_score)
	print ("Classification Score : %f" % clf_score)
	print ("Total Score : %f" % (loc_score + clf_score))