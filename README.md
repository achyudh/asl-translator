# Translation of American Sign Language Gestures
This project utilizes and compares various models based on CNNs, RFCs, and SVCs to recognise hand gestures and translate sign language images and videos into text.

## Approach
- Tuned support vector, random forest and gradient boosting classifiers to build an ensemble learning model based on hard voting
- Adopted a moving window based approach and extracted Histogram of Gradients, Local Binary Patterns, and DAISY features from the images in order to improve the accuracy scores
- Utilized techniques like hard negative mining and non-maximal suppression for localization of the hand performing the gesture
 
## Results
Our approach achieved accuracy of greater than 99% on localization (Metric: IoU should be larger than 0.50) and 96.8% on top-5 classification of test data. The dataset contained a total of 26 classes with around 5000 images.
 
## Prerequisites
I would recommend using the Anaconda package with at least Python 3.5. The implementation made use of the classifiers and feature selection algorithms implemented in the Scikit-learn library for Python, and Scikit-image library for handling images. Other dependencies like Numpy will be taken care of during the installation of Scikit-learn and Scikit-image.
 
## Contributing
When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change. Ensure any install or build dependencies are removed before the end of the layer when doing a build. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.

## License
This project is licensed under the MIT License. Please see the LICENSE.md file for details.

## Acknowledgments
I worked with Edwin Mascarenhas for the implementation and would like to mention him for his contribution. I thank Sanket Gupte for his guidance and helping us with some of the ideas implemented in this project. 
