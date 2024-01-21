# Lecture 1: Jan17 2024

## Basic of image
- Every pixel of an image show the intensity of image, RBG
- Every image has Height Width or depth.
- 3 diffent color map supperimpose to show the color image.
![Alt text](https://www.theclickreader.com/wp-content/uploads/2020/08/color-channels-RGB.jpg)
- each point is called pixel and each pixel has 3 color(RBG)
- every color image are in metrix(3x3x3)
- every color is made up of RGB intensity 
![Alt text](https://qph.cf2.quoracdn.net/main-qimg-b812e8fc338dc7706e3767dc02cc89ad)
- higher the pixel better the image
- image has structure data, you will never found missing value, you will never find string value here, it has better structure and range is between 0 to 255.

## How our brain understand this is tree, sun or river?
> Because of the cluster of the yellow and pixel location of the sun

## what is the information in an image:
> pixel density or veriation of pixel density at different location in an image. 

## Why ANN cannot work for images?
 - We take an image, Flatten it and pass it to neural network. but it won't work, suppose we have an image(object sun, river, tree).
 - new pixel related to tree should combine in seperate manner, pixel related to river should combine in seperate manner and pixel related to sun should combine in a different manner.
 - we can not combine everything and pass it to neural network.
 - In mnist Handwritten dataset we can apply ANN because everything is black except the digit.

## convaluation process
 - it is a feature extraction process, 
## Concept of keranl
 - we want a machanism where i can extract the features at local level, that is where a concept of windows or kernal comes in.
 - A keranl is basically a metrix of certain weight much lower dimension then the image, let image maybe 100x100 and kernal will be 3x3.
 - A keranl is weight metrics, kernal will get supperimpose on an image, image have some value the keranl have some value.
 - we apply .dot product will happen, and pass it to activation and a neuron is generated.

![Alt text](https://miro.medium.com/v2/resize:fit:1358/1*D6iRfzDkz-sEzyjYoVZ73w.gif) 

### 1. Edge Filter(Vertical edge detector)
![Alt text](https://media.licdn.com/dms/image/C5112AQHeSoRNyzPq0w/article-cover_image-shrink_720_1280/0/1520791281138?e=1710979200&v=beta&t=5b9-zyO_TYepSJGVqcefYr-xx6-3BsLjLFiXTkgAZhM) 
 - left size - white region, middle - dark region, right side - darker region
 - when there is same pixel then there is not edge, if they have differnt pixel then it is edge.
 - diffence in the pixel on right hand side and left hand size.
 - when we have this kernal and convalute to the image it mean we are search for edge in that image.
 - **Now question is that how to make cat detector, or circle detection, we do so by learning the weight**

## is it possible to find a car, or a tree from single keranl?
 - No, we need more then one keranl to find the car from the image.
 - We make a learnable kernal and convolute to an imaga and create the feature map.
 - if we have 100 of kernal then it create 100 of feature map.


## Feature Map
- Different Kernel would have learn different features from the image and presented it in the feature maps
- so basically i took the image of the sun, tree, and car, and then using the kernels i learnt different featueres of the sun, tree and car and presented them in feature maps.
- neural network is basically going to learn the weights of the kernels.

- **Dot product happen 3 channel of color with kernal**

## what is information in an image?
- Varision in the pixel intensity is the inforamtion of an image.
- certain pixel are club together or present together that form an object.

## what happend when we flatten an image?
- when we flatten the image complelty and crate the 1 dimenstitional vector that vector, that vector the problem is that we loos locational or speical corelational.

## Feature Extraction
- if we do dot product it will extract feature.
- suppose if my kernal contain plus sign if it do the dot product, we will get higher value where the plus sign is avalable else lower value at feature map.

## Important point related to kernal
 - 1x1 is very small kernal it can not detect anything
 - 2x2 is do not follow symetric
 - 3x3 follow synmetic that is why we are taking smalles kernal as 3x3.

## if we have large size kernal then what is the issue?
 - There is chances that sun and bird both can capture by single neuron.
 - risk of including other object in that location.

## if large kernal is issue then small kernal may also have problem?
 - for example, one 3x3 kernal can not be detect whole sun.
 - using the 3x3 kernal may not capture our sun properly.
 - there is not problem because one kernal create one feature map which is pass by one neuron.
 - one neuron can learn left part of sun, one neuron learn right part of sun, one neuron learn bottom part of neuron. so there is no issue. but if we take larger kernal size then there is an issue that both sun and tree capture by single neuron, and that neuron get confuse which output to product sun or tree.
 - so small size kernal is not has big risk.
 - small size kernal is less infomation but there is not risk of capturing nebouring object.

## where we are using the large size kernal. 
 - yet to answer

## why kernal has to be symetric not square
 - yet to answer

## What is the dimension of feature map
 - if you have 100x100 pixel image(RGB) and has 100 of 3x3 kernal, the output of feature map. we will get with strive=1 is find out using the formula
 - H' = [(H-f)s]+1
 - W' = [(W-f)s]+1
 - depth = no of filter we have applied
## now, how many parameter the above configuration has.
 - 28 because, for each 3x3 karnal we have 9 weight and we have 100 of them, which is (3x3x3)+1(bias)
 - Number of learnable parameter is independent of size of image.
 - it is depended upon the size of kernal

 ![Alt text](https://qph.cf2.quoracdn.net/main-qimg-b662a8fc3be57f76c708c171fcf29960)

 ## Max Pool
  - max pool is also a kind of convalution operation only with some filter size, this operation took max out of it. 
  - this is a clean up process, we are saying just give me max out of it.
  - when we train a kernal which can learn only tree(suppose) object from an image, there is high chances that that kernal takes some part of sun(suppose) so for that we need some clean up process, this clean up process is called max pooling.
  - suppose we have 28x28x3 image and we apply dot product with 3x3x100 kernal it will generate 26x26x100 feature map.
  - we already know that dring colvalution it take the some part of neabouring object so this max pooling remove that part and takes the max out of it.
  - ![Alt text](image-2.png)
  - after this i would have clean feature map.

## Convolutional Neural Networks
 - Apply Convalution to extract feature and max pooling to clean up feature, repect twice thrise. max pooling get raid of unnessary neuron.
 - will do the **global max** at the end, which will give me most important feature of the each feature map.
 - using the global max, each map talk about the most import part of the tree in the feature map.
 - then we pass it to dense layer to predict the tree, that is what convoltution articture looks like.
![Alt text](image-3.png)







