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

**Dot product happen 3 channel of color with kernal**