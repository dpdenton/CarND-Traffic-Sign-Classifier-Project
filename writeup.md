#**Traffic Sign Recognition Writeup**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/dpdenton/CarND-Traffic-Sign-Classifier-Project)

###Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the total count of each class:

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Traffic-Sign-Classifier-Project/master/plots/class_freqency.png)

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I converted the images to grayscale help the network detect edges.

I created additional data for classes where the total count was less than the max(totalCount). The number of additional examples generated was dependent on the total count of the class, defined by the formula *c \. floor(max(C) / c - 1)* where *C* is a list of class counts and *c* is the class count being evaluated.

These samples were randomly perturbed in position ([-2,2] pixels), as per [LeCun's paper](http://yann.lecun.com/exdb/pubalis/pdf/sermanet-ijcnn-11.pdf) to prevent the network overfitting a specific image representation.

As a last step, I normalized the image data because this improves the performance of the network and reduces math error.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16     									|
| RELU					|
| Max pooling			| 2x2 stride,  outputs 14x14x6
| Flatten				| outputs 400
| Dropout				| keep_prob: 0.5
| Fully connected		| Input = 400. Output = 120
| RELU					|
| Fully connected		| Input = 120. Output = 84
| RELU					|
| Fully connected		| Input = 84. Output = 43
| Softmax				|

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The network is based upon the [Yann LeCun's Network](http://yann.lecun.com/exdb/pubalis/pdf/sermanet-ijcnn-11.pdf)

By generating the additional data, increasing the epochs to 50 and reducing the learning rate to 0.0006, the validation accuracy reached ~ 0.935.

I didn't change the architecture of the network as it was performing around 0.94 validation without any change.

The network was overfitting because the training accuracy was 1.0 but increasing the number of epochs decreased the accuracy, so I applied dropout with a keep\_prob of 0.5, which improve the validation accuracy to around 0.96

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.966
* test set accuracy of 0.950

I chose LeNet architecture because it's proven to have a good baseline for this very task.

The network allows the classifier to use not just high-level features, but also pooled lowlevel features.

 The test accuracy set was 0.95, compared to the validation accuracy of 0.966 and the training of 0.99. This suggests the network has trained reasonably well, however the 0.033 difference between the train and valid accuracy still suggests the network is overfitting. This is supported by the fact the test accuracy is lower than the valid.. it's trained well for the valid set, but it not able to generalise that learning over to the test set. For a well trained network I would like to see no difference between the test and validation accuracy (whilst still trying to achieve the highest accuracy possible) as this is a sign the network can generalise its perfectly.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Traffic-Sign-Classifier-Project/master/downloads/no_passing.jpg)![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Traffic-Sign-Classifier-Project/master/downloads/slippery_road.jpg)![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Traffic-Sign-Classifier-Project/master/downloads/speed_120.jpg)![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Traffic-Sign-Classifier-Project/master/downloads/yield.jpg)![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Traffic-Sign-Classifier-Project/master/downloads/stop.jpg)

I manually pre-processed the images here. I was struggling to find any images on the web without watermarks and where the sign was sufficiently large.

Either there were watermarks on the image or the sign was so small relative to the size of the image, if I'd resized it down to 32x32 the sign would have represented a single pixel making it impossible to classify. I guess this might have been part of the challenge in retrospect!

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| No entry    			| No entry   									|
| Slippery road  		| Slippery road 										|
| Yield					| Yield											|
| Speed limit (120km/h)	| Speed limit (120km/h)					 				|
| No passing			| No passing      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the **Slippery road** sign the top 5 probabilities were:

* 1.00000000e+00,
* 1.62315352e-08
* 9.46953627e-10
* 1.13089054e-14
* 1.50508755e-16

And corresponding classifcations: [23, 30, 25, 11, 20]

I found this quite concerning as, aside from it not equalling 1, I'd say the network is overly confident of its prediction.

If the network had produced a training, valid and test accuracy of 100% then I wouldn't have an issue with this result.

However the network did not even classify every **Slippery road** sign correctly,  so to a assign a probability of 1, despite it getting the prediction correct, suggest the network is over-fitting and might struggle to generalise if presented with more difficult input images.

This prediction is similar in the subsequent images:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.     				| Slippery road 										|
| 1.					| Yield											|
| .95	      			| Speed limit (120km/h)						 				|
| 1.				    | No passing

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

