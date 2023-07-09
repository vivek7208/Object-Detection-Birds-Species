# 🐦 Object Detection for Bird Species 🐦

[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/vivek7208/Object-Detection-Birds-Species/blob/main/object_detection_birds.ipynb)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![Preview in nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/vivek7208/Object-Detection-Birds-Species/blob/main/object_detection_birds.ipynb)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vivek7208/Object-Detection-Birds-Species/blob/main/object_detection_birds.ipynb)

## 🎓 Introduction
This project demonstrates the use of Amazon SageMaker's object detection model to identify bird species 🐤🦆🦅. The project uses the Caltech Birds (CUB 200 2011) dataset and the Single Shot Multi-Box Detection (SSD) algorithm. The process includes data preparation, model training, and deployment on Amazon SageMaker. This README provides a comprehensive walkthrough of the steps involved in the project.

## 📚 Dataset
The project utilizes the Caltech Birds (CUB 200 2011) dataset, which contains 11,788 images across 200 bird species. Each species is represented by approximately 60 images. The images vary in size, typically around 350 pixels by 500 pixels. This dataset provides bounding boxes for each image and annotations of bird parts, but does not include image size data. 

The data preparation process includes downloading the dataset from Caltech, unpacking the dataset, and exploring the data. The images in the dataset are divided into individual named (numbered) folders, which effectively label the images for supervised learning. 

![image](https://github.com/vivek7208/Object-Detection-Birds-Species/assets/65945306/1f7056c7-23f8-4308-a3f8-c60ba52d306b)


## 📋 Data Preparation
The data preparation process starts with understanding the dataset and setting certain parameters such as `SAMPLE_ONLY`, which determines whether the model trains on a handful of species or the entire dataset 🌍. 

Next, the RecordIO files are generated. The RecordIO format requires bounding box dimensions to be defined in terms relative to the image size. The images are visited, and the height and width are extracted for subsequent use. 

Following this, list files for producing RecordIO files are generated. These list files include one row for each image with bounding box data and a class label. The bounding box dimensions are converted from absolute to relative dimensions based on image size, and the class IDs are adjusted to be zero-based. 

Lastly, the data is converted into RecordIO format and uploaded to the S3 bucket in multiple channels 🚀. These channels differentiate the types of data provided to the algorithm. 

## 🧠 Model
The object detection model used in this project is based on the Single Shot Multi-Box Detection (SSD) algorithm. This algorithm uses a base network, which is typically a VGG or a ResNet. The Amazon SageMaker object detection algorithm supports VGG-16 and ResNet-50. 

The training process begins by defining an output location in S3, where the model artifacts will be placed upon completion of the training. The URI to the Amazon SageMaker Object Detection docker image is also obtained.

The SSD model is known for its accuracy and speed, making it a popular choice for object detection tasks. The model essentially views the problem of object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A set of default bounding boxes over different aspect ratios and scales per feature map location is used.

## ⚙️ Hyperparameters
Hyperparameters help configure the training job. They include:

- `num_classes`: The number of output classes for the new dataset.
- `num_training_samples`: The total number of training samples.
- `epochs`: The maximum number of iterations for the optimizer during model training.
- `learning_rate`: The learning rate for the optimizer during model training.
- `mini_batch_size`: The number of mini batch size used for the single-machine multi-GPU/cpu training.
- `overlap_threshold`: The overlap threshold for deciding true/false positives.
- `nms_threshold`: The overlap threshold for Non-Maximum Suppression (NMS).
- `image_shape`: The input image dimensions,'300' is used for training on Pascal VOC dataset, and '512' is used for training on MS COCO dataset.
- `label_width`: The number of labels for the object detection task.
- `base_network`: The base network for the SSD model.

The hyperparameters are set up, and the data channels are linked with the algorithm. The training job is then submitted, and the progress is monitored. The provisioning and data downloading may take some time, depending on the size of the data.

![image](https://github.com/vivek7208/Object-Detection-Birds-Species/assets/65945306/8211ae56-4c96-42c3-aedd-0e707164b8e4)


## 🌐 Hosting the Model
After the model is trained, it is deployed as an Amazon SageMaker real-time hosted endpoint. This allows making predictions (or inferences) from the model. The endpoint deployment is accomplished with a single line of code calling the `deploy` method. 

## 🧪 Testing the Model
The model is tested by making predictions using the hosted model and visualizing the results. The results of a call to the inference endpoint include a confidence score for each detected object. Low-confidence predictions are typically not visualized.

![image](https://github.com/vivek7208/Object-Detection-Birds-Species/assets/65945306/fcbd024d-7ebc-4392-a4c7-c3a30da96202)
![image](https://github.com/vivek7208/Object-Detection-Birds-Species/assets/65945306/0225e6f5-9b31-4f02-80f8-6cbdd7b6aa2c)


## 🎯 Model Improvement
The notebook provides a method to improve the model by flipping the images horizontally and retraining the model with the expanded dataset. This data augmentation strategy can help improve the model's performance.

![image](https://github.com/vivek7208/Object-Detection-Birds-Species/assets/65945306/e71d447b-10a8-4a77-a5e1-04c4a8766351)


## 🧹 Cleanup
Finally, the SageMaker endpoint is deleted to avoid any unnecessary charges 💰.

This README serves as a comprehensive guide to the project. For more details, please refer to the Jupyter Notebook, which includes code, visualizations, and in-depth explanations for each step. 📘🔍
