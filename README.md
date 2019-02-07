# CIFAR-10

![](/images/cifar-10.png)

## Overview
In this document I will explain in details my approach of solving the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) challenge.
I will describe in details the problem, the solution, and describe the way I choose the different parameters.
I will begin with the bottom line: Using my own personal laptop machine which is a Windows-7 64-bit with 8GB of RAM, and Intel Core i5-5300U CPU of 2.30 GHZ, I have successfully matched 70.41% of the images in about 53 minutes.

## The problem
CIFAR-10 is a dataset consist of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The test images are composed of 1000 randomly selected images from each class. The training images contains 5,000 images from each class in random order.
The goal of this competition, is to build an algorithm which takes a random image and classify it to one of the 10 classes with as high as possible accuracy.

## The Algorithm
### Overview
Basically we are talking about a classifier algorithm. This algorithm should use the training images to build a classifier model, and then apply the test images on this model to test how good this model really is.
The following chart shows the building block of the algorithm:

![](/images/algorithm.png)

#### Training Phase
1. **Training Image:** The algorithm start with a set of 50,000 classified training images.
2. **Preprocessing:** These images are send to the preprocessing phase, in which they are prepared for the next phases. For example cleaning the images from noise, so that the noise will not affect the next phases.
3. **Feature Extraction:** Now, each image is ready for feature extraction phase, in which it is represented as a vector
4. **Dimensionality Reduction:** Once the images are converted into vectors, the algorithm reduce the dimension of these vectors. This dimension reduction is done under the assumption that similar images (from the same class) will have similar (or close) vectors.
5. **Learning Algorithm:** Now, these reduced vectors are ready to be sent to the learning algorithm, which receives these vectors of images a long with the label of each image, and calculate the classifier model. This model will be used later for the test images.

#### Testing Phase
Once the model classifier is ready, it can used by the test images. 
These test images are going through the same pipeline of: preprocessing, feature extraction, and dimension reduction. 
Then they are ready to be sent to the model classifier created in phase 5 above, to be classified. 
The solution is than compared to the real classes of the test images and the accuracy level and confusion matrix are calculated. 

### Reading The Images
1. **Input:** The 5 training files from the given dataset
2. **Output:** 2 arrays as follows:
     - Array of the 50,000 training images
     - Array of the 50,000 labels

### Preprocessing
At the beginning I thought that it is important to filter the noise out of these images, so the feature extraction will not be effected by noise. However, my tests showed that any preprocessing method I have tested only decreased the success rate. Therefore I have decided not to touch the image in this phase. I have tried methods like:
- Gaussian filter
- Converting the images to gray scale
- Using the image received after applying the “Canny Edge detector” algorithm (tried with several values of “min” and “max” thresholds)
- Using the “Harris corner detector”, and feed these corners to the next phase.

I guess that since these images are small, every pixel counts, and since in any of the above method, there is a certain amount of information lost, this leads to degradation in the success rate. Therefore, I have decided not to implement this phase at all.

### Feature Extraction
- Input: Array of 50,000 images
- Output: Array of 50,000 vectors of 5832 components each, received by HOG algorithm.

The goal of this phase, is to be able to convert an image, into a vector which holds the important information (features) of the image. In our case, since the images are very small, and many features typically used to identify an object may be highly blurred, we can use only object outlines to distinguish between classes of object. Histogram of Oriented Gradients (HOG descriptors) do exactly that.
In short (without getting into the details of HOG), HOG descriptor is computed by calculating image gradients that capture the image contour and silhouette information. Gradient information is pooled into a vector, which is a histogram of orientations.
HOG calculation requires tuning several parameters like cellSize, blockSize, blockStride, nbins and signedGradient. After visualizing the HOG calculation for several values, I have found out that the best result are achieved by using the following parameters:
- cellSize = 4
- blockSize = 12
- blockStride = 4
- nbins=18
- signedGradient = True

Using these parameter every images is converted into a vector of 5,832 components. This vector can be viewed as the vector representation of the original image, which hold the object’s contour information.

### Dimensionality Redution
- **Input:** Array of 50,000 HOG vectors of 5,832 components each
- **Output:** Array of 50,000 vectors of 3,000 component each, received by PCA transform.

Dimensionality reduction is the process of reducing the number of dimension in a given dataset. The reduction is done by removing the correlation between the components in the dataset, leaving only the components with high variance.
PCA – Principal Component Analysis – is a statistical procedure which allows us to do exactly that. In other words PCA is a procedure which allow us to transform a vector of m dimension to a vector of d dimension where d<<m, making sure that the correlation between the various components is minimized.
In my implementation I have used the OpenCV implementation of PCA. The input to the PCA algorithm is the 5832-dimension HOG vectors calculated in the previous step. Where the goal is to reduce the dimension of the HOG vectors to a smaller dimension, making sure that we capture most of the information, and minimize the correlation between the dimensions. But, what is that lower dimension? Is it 10? 100? 1000?
Normally, when using PCA, the value of the lower dimension has to be given to the algorithm. If the dimension is not given, all the components are stored (no reduction is done), and the PCA instance holds a special array called **explained_variance_** which holds the amount of variance for each component. In addition **explained_variance_ratio_** holds the percentages of variance for each component.
I have executed this algorithm using the HOG vectors and plotted the output using:
```
np.cumsum(pcs.explained_variance_ration_)
```
And got the following chart:

![](/images/pca.png)

The meaning of this chart, is that most of the information is held in the first 2000-2500 components. I have tried with several numbers (1500, 2000, 2500 and 3000) and found that 3000 component gave the best results. So in this phase, we took a 5,832 dimension vectors, and transformed them into a 3,000 dimension vectors.

### SVM – Support Vector Machine
- **Input:** An array of 50,000 vectors reduced by PCA to 3000 components each, and an array of 50,000 corresponding labels.
- **Output:** A multi-class SVM classifier.

Up until now, I have showed how I take the training images, extract from each one of them the HOG descriptors, and reduce each one from a dimension of 5,832 components to 3000 components, using PCA.
Now it’s time for the “real-deal” using multiclass SVM classifier. Feeding the algorithm with the 50,000 vectors received as input to this phase, and the 50,000 labels, the algorithm build a multiclass classifier. This classifier can be saved as a file to be used later on the test data. 

### Testing The Classifir
Now that the classifier is ready, we can go into the testing phase. In this phase, we load the test images, and send them through the same pipeline of creating HOG descriptors, reducing their dimensions, and sent to the classifier to be predicted according the model created in the previous step.
The prediction is than compared to the real labels, to calculate the confusion matrix, and the accuracy level.
