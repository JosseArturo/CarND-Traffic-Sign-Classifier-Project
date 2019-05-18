# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

Steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Apply softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: training_set.PNG "Visualization"
[image2]: dataS.PNG "DataSet"
[image3]: ./New/60_kmh.jpg "60Km"
[image4]: ./New/left_turn.jpeg "Left Turn"
[image5]: ./New/road_work.jpg "Road Work"
[image6]: ./New/stop_sign.jpg "Stop Sign"
[image7]: ./New/yield_sign.jpg "Yield"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. The submission includes the project code.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. The analysis was done using numpy library

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed.Se observa un Data

![Visualization][image1]

An unbalanced dataset is observed. This can be an opportunity in the future. Data augmentation can be one of the best ways to guarantee a model with good generalization capabilities

Some images were plotted to see the quality of the input images

![DataSet][image2]

In these images it is observed that the dataset is composed of low resolution images with variable rotations and scales. It is observed that some traffic signs have a specific color. It is decided then not to use color transformations in order to use this information that may be relevant

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques?.

Analyzing the problem in detail, it is decided not to generate modifications in the data set, this with the aim of seeking to exploit the model as much as possible. The only processing consists of an optional normalization that can improve the model.


Normalization guarantees that all the features are scaled to the same rank, helping the decendent gradient to converge quickly.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 	| Convolution 3x3     	| 1x1 stride, valid padding, outputs 12x12x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x32 
| Flatten				| input 6x6x32, outputs 1152												|			
| Fully connected		|input 1152, output 64	         									|
|Dropout				| 25%        									|
| Fully connected		|input 64, output 43	         									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model the Lenet architecture was used. So, for the loss function was Cross Entropy function used and the optimazer was an ADA optimizer with a learning rate of 0.0005.
The epochs were 70 with a batch size of 32.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* Train Set Accuracy = 99.9%
* Validation Set Accuracy of 97.5%  
* Test Set Accuracy of 96.6%

If an iterative approach was chosen:


Based on the Lenet architecture, different changes were made to increase the performance of the model.
The dimensions of the kernels were changed and a convolutional layer was added. Also, the depth in the results of the convolutional layers was maintained (32), since an increase in this depth did not mean any improvement in the model.

A fully connected layer was eliminated from the end of the network and the dropout was reduced, the reason for this were the limitations of the testing dataset. The big motivation to avoid data augmentation was to exploit the model as much as possible and find the architecture for a data set training with great limitations. And it could be found that a bigger dropout was a bad option for a very limited dataset and more than two fully connected layers at the end were to claimant for this training data set. 

The difficulty to use the original Lenet architecture was the special dataset for the trafic sign. It is an unbalanced dataset with a poor number of samples. 
With the original architecture the model was had a good result but enoguh, in terms of performance, so the model was underfitting for the purpose of the final accuracy needed.
The over fitting was not a presented case in all the probes done, because the number of the batch, epochs and rate was controled to avoided. The batch was not to small and the epochs were not to many. The rate was getting lower to help the optimizer to arrive to a better resut, doing the model slower but more precise

One big conclusion of this execise is the importance of the Data set. It is necesary 

It is necessary to understand the behavior of my images and the features that I can extract from them. It is important to understand that without a better dataset the model will not achieve a better performance than the one obtained (97%). Also, considering the number of classes and the complexity in the images, the convolutional layers must be added or removed, in order to abstract enough features fro the image.
 
Finally, and even having large limitations in the dataset, a model was achieved that responds appropriately to the test data set. This verifies that the model has not overfitting.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:


![60Km][image3]![Left Turn][image4] ![Road Work][image5] 
![Stop Sign][image6] ![Yield][image7]


These images might be difficult to classify because of the background of the trafic sign

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Test Accuracy = 100%

The model classified 5 of the 5 traffic signs,accuracy of 100%.


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 60 Km      		| 60 Km   									| 
| Stop     			| Stop 										|
|Road work			|Road work											|
| Turn Left	      		|  Turn left ahead					 				|
|Yield			|Yield      							|


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

In general, it is verified that the model correctly finds all the images. In the same way it is observed that the probability that the images found belong to another class are so small that they are almost zero percent.


It is presented below for each class the 5 classes that were probably closer to the final decision of the classifier.

60_kmh.jpg:
Speed limit (60km/h): 100.00%
No passing: 0.00%
No passing for vehicles over 3.5 metric tons: 0.00%
Ahead only: 0.00%
Speed limit (50km/h): 0.00%

stop_sign.jpg:
Stop: 100.00%
Speed limit (20km/h): 0.00%
Speed limit (30km/h): 0.00%
Speed limit (50km/h): 0.00%
Speed limit (60km/h): 0.00%

road_work.jpg:
Road work: 100.00%
General caution: 0.00%
Speed limit (30km/h): 0.00%
Traffic signals: 0.00%
Bicycles crossing: 0.00%

left_turn.jpeg:
Turn left ahead: 100.00%
Keep right: 0.00%
Speed limit (20km/h): 0.00%
Speed limit (30km/h): 0.00%
Speed limit (50km/h): 0.00%

yield_sign.jpg:
Yield: 100.00%
Priority road: 0.00%
Speed limit (60km/h): 0.00%
No passing: 0.00%
No entry: 0.00%