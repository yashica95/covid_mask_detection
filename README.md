# Use OpenCV and TensorFlow to detect if the person is wearing a mask 

## Project Overview:
The projects aims to detect if a person is wearing mask or not on a live video. The project can be useful in offices and public spaces to detect if everyone is wearing a mask and adhering to the WHO guidelines. It's a humble request to everyone to follow the guidelines and maintain social distancing to keep yourself safe and healthy. 

## The Process:
1. **Training**: Load face mask detection dataset, train a model in Tensorflow using transfer learning, and then serializing the face mask detector to disk
2. **Deployment**: Load the mask detector, perform face detection with OpenCV, and then classify each face as with_mask or without_mask

## Dataset Source:
We are using a small subset of dataset created by Prajna Bhandary. This subset of dataset consists of 194 images belonging to two classes: 
- with_mask: 97 images
- without_mask: 97 images

## Requirements:
- tensorflow
- imutils
- numpy
- sklearn
- matplotlib

## Model Performance:
The model performed well with our small dataset of only 194 images and gave accuracy of **97.44%** on Validation data. The dataset can be expanded to include more realistic images of people wearing masks in different situation, with different poses. 

![](https://github.com/yashica95/covid_mask_detection/blob/master/model_performance.png)


## Acknowledgment:
The code has been taken from [this blog](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/) with slight changes in the model. I have changed the dropout layer to only drop 20% of the data rather than 50% data since we have a very small dataset. Also, I have decreased the number of epochs to avoid overfitting and still get same validation accuracy in less time.  
