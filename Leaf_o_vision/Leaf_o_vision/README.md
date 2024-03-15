# **LeafVision Disease Detection**

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org)
[![streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io/)

An end-to-end Deep Learning project to detect diseases that attack corn leaves

## **Problem Statement**
The plant diseases threaten global food security and smallholder farmers whose livelihoods depend mainly on agriculture and healthy crops. In developing countries, smallholder farmers produce more than 80% of the agricultural production, and reports indicate that more than fifty percent loss in crops due to pests and diseases. The world population is expected to grow to more than 9.7 billion by 2050, making food security a major concern in the upcoming years. Hence, rapid and accurate methods of identifying plant diseases are needed to take the appropriate measures.

This Streamlit App utilizes a Deep Learning model to detect diseases that affect the corn leaves, based on digital images.


## **Data Preparation**

The plant-village dataset contains 39 different classes of plant leaves (healthy and unhealthy) and background images(61,486 in total)(Geetharamani & Pandian,2019). In this project, we used the version without augmentation and just used the corn dataset, which contained four different classes (Blight, Common Rust, Gray Leaf Spot, and Healthy).

Data preprocessing steps:

- Data normalization[0,1]
- Data augmentation using several techniques such as:
  - image flipping
  - zoom
  - shear
  - width and height shift
  - image rotation
  - Image brightness range
  - Featurewise center
  - Featurewise std normalization

[Dataset Link](https://data.mendeley.com/datasets/tywbtsjrjv/1)

## Modelling
In this project, I tested the following Convolutional Neural Network Architecture by transfer learning method:

 - MobileNetV2 (Pretrained with imagenet images)

Fine Tune model Performance: 93% (accuracy score metric)

## **Deploy**
The Streamlit app was deployed on Streamlit Cloud
