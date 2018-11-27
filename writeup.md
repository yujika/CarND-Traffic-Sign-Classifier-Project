# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visualize.png "Visualization"
[image2]: ./grayscale.png "Grayscaling"

[image4]: ./test_0.png "Traffic Sign 1"
[image5]: ./test_1.png "Traffic Sign 2"
[image6]: ./test_2.png "Traffic Sign 3"
[image7]: ./test_3.png "Traffic Sign 4"
[image8]: ./test_4.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/yujika/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the array shape to get set size and pandas library to get number of unique labels of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x1
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![visualize dataset][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

- Converted 32x32x3 RGB image into 32x32x1 gray scale image to reduce number of input data
  - gray scale image ( 0 - 255 ) then normalized to ( -0.5 - 0.5 ) for optimizer to work efficiently

- Below is an example of a traffic sign images after and before grayscaling.

![alt text][image2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 normalized gray scale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling   	   	| 2x2 kernel 2x2 stride,  outputs 14x14x6	    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling   	   	| 2x2 kernel 2x2 stride,  outputs 5x5x16	    |
| Drop out              | drop out                                      |
| Flatten               | outputs 400                                   |
| Fully connected		| inputs 400 outputs 120                 		|
| RELU					|												|
| Fully connected		| inputs 120 outputs 84                 		|
| RELU					|												|
| Fully connected		| inputs 84 outputs 43                 		|
| Softmax				| one hot encoding								|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as ing rate.

To train the model, I used the following
- Added drop out layer with 0.35 probability to prevent over fitting, because default LeNet works almost OK but it was over fitted.
- Learning rate = 0.0001 after comparing accuracies between rate 0.01, 0.001, 0.0001 and so on.
- optimizer is Adam optimizer because it is default optimizer of LeNet and I think it is OK.
- batch size is 256
- number of epoch is 200. It seems that 100 epoch is OK, but accuracy go up a littele bit even after 100, so I decided the number of epoch.
- tried out some sigma values and I decided sigma = 0.2 is OK.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.961
* test set accuracy of 0.955

I chose iterative approach:
* What was the first architecture that was tried and why was it chosen?
  * I chose LeNet, because LeNet is only practical CNN I learned 
* What were some problems with the initial architecture?
  * Training set accuracy is higher than target accuracy of 0.93
  * But Validation set accuracy is lower than 0.93 ( over fitting )
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * Because LeNet shows over fitting ( high accuracy on the training set but low accuracy on the validation set ), I chose dropout to prevent over fitting.
* Which parameters were tuned? How were they adjusted and why?
  * learning rate, because it is the most important parameter
  * dropout rate, because I think higher dropout rate prevents over fitting more. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  * What architecture was chosen?
    * LeNet
    * A dropout layer added after second convolution layer to mitigate over fitting 
  * Why did you believe it would be relevant to the traffic sign application?
    * It classified hand writing number very well
  * How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    * It achieved more than 0.93 accuracy on validation set
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

* 4th image is speed limit 70km/h but very dark and might be diffcult to classify
* 5th image is speed limit 80km/h but very dark and might be difficult to classify

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 20km/h    | Speed limit 20km/h		| 
| Speed limit 30km/h    | Speed limit 30km/h				|
| Speed limit 50km/h    | Speed limit 50km/h			|
| Speed limit 60km/h	| Speed limit 60km/h		 				|
| Speed limit 70km/h	| Speed limit 70km/h      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 1000%. 
Initially I mistakenly feed reversed value image ( black=1.0 and white =-1.0 ), with that data, network showed very bad result.
But after I corrected image data, network did good job as test accuracy reported ( 0.95 ).

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image - Speed limit 20km/h, predicted as speed imit 20km/h in very high probability of 0.95. very good result and much to test accuracy.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .950         			| Speed limit 20km/h							| 
| .049    				| Speed limit 30km/h							|
| .001					| Speed limit 60km/h							|


For the second image - Speed limit 30km/h, predicted with almost 1.0 probability. Better than ist image.
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99996         			| Speed limit 30km/h							| 
| .00003     				| Speed limit 70km/h							|
| .00001					| Speed limit 40km/h							|

For the 3rd image - Speed limit 50km/h, predicted correctly with almost 1.0 probabirlity.
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999996     			| Speed limit 50km/h							| 
| .000003  				| Speed limit 80km/h						|
| .000001				| Speed limit 30km/h					|

For the 4th image - Speed limit 60km/h - predicted correctly with almost 1.0 probability
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999994     			| Speed limit 50km/h						| 
| .0000005     			| Speed limit 80km/h							|
| .0000001				| Speed limit 30km/h						|

For the 5th image - speed limit 70km/h, - predicted correctly with almost 1.0 probability
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .999970      			| Speed limit 70km/h						| 
| .000025  				| Speed limit 30km/h						|
| .000002				| Speed limit 120km/h						|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
I showed 2nd convolution netowrk. But didn't make sense...



