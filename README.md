[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)]()
[![forthebadge](images/powered-by-aws.svg)]()
[![forthebadge](images/made-with-flask.svg)]()

<h1 align="center">Toxic Comment Classification using Flask & AWS 🔍</h1>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]()
  [![Open Source Love png1](https://badges.frapsoft.com/os/v1/open-source.png?v=103)]()
  [![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)]()

</div>

---

<p align="center"> This is a toxic comment classifier web application that uses a trained Logistic Regression model to predict the toxicity levels of a given text input.
</p>

<h3> Link to the web app: 👉🏻 <a href="http://ec2-18-117-78-151.us-east-2.compute.amazonaws.com:8080/">Toxic Comment Classifier</a></h3>

<em>Disclaimer: the dataset for this project contains text that may be considered profane, vulgar, or offensive.</em>

## 📝 Table of Contents

- [🧐 About](#about)
- [🎯 Getting Started](#getting_started)
- [📊 Dataset Overview](#data-overview)
- [🧠 Model Building](#machine-learning-model)
- [🎈 Usage](#usage)
- [🚀 Deployment](#deployment)
- [🌟 Support](#support)

## 🧐 About <a name = "about"></a>

This is a multi-label classification problem where the given input is a text comment and the output is list of the toxicity level it belongs to. 

The input text data needs to be cleaned and pre-processed for it to be useful for the Machine Learning model.

## 📊 Dataset Overview <a name="data-overview"></a>

The dataset for this problem was taken from competetiion hosted by <a href="https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data">Jigsaw</a> on Kaggle.

For preprocessing of the input data and text vectorization, both word and char based TF-IDF vectorizer's output are used as inputs to the model for better performance and minimum loss of input features.

The different types of target labels present are: ```toxic, severe-toxic, obscene, threat, insult and identity hate```.

### Click to view 👇:

[![forthebadge](images/solution-exploratory-data-analysis.svg)](https://github.com/vipul-shinde/toxic-comment-classification/blob/main/notebooks/01-eda-and-data-cleaning.ipynb)

## 🧠 Model Building <a name="machine-learning-model">

For building the classifier, we have used Logistic Regression and treated the multi-label problem as a binary problem. The reason for this approach instead of a OneVsRest Classifier is because of better model performance when the problem is treated as a binary one. 

Since the data is unbalanced, just accuary in itself cannot be considered as a strong evaluater, therefore we have used F1-score along with it to evaluate the model.

Here are the results on validation and test datasets:

```
Validation Results 👇🏻

Validation Accuracy: 0.9828502793879577
Validation F1-Score: 0.9811947440446507
```

```
Test Results 👇🏻

Test Accuracy: 0.9752805651942854
Test F1-Score: 0.9747181660736461
```

### Click to view 👇:

[![forthebadge](images/solution-exploratory-data-analysis.svg)](https://github.com/vipul-shinde/toxic-comment-classification/blob/main/notebooks/02-model-building.ipynb)

## 🎯 Getting Started <a name = "getting started"></a>

*Project Structure:*

```
Volume serial number is D8B2-80F9
D:.
├───data
│   ├───cleaned-data
│   └───raw-data
├───images
├───models
├───notebooks
├───static
│   └───css
├───templates
└───__pycache__
```

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

```
Flask==1.1.2
joblib==1.0.1
nltk==3.6.1
numpy==1.20.1
pandas==1.2.4
scikit_learn==1.0.2
scipy==1.6.2
swifter==1.0.9
```

### Installing

Use [miniconda](https://docs.conda.io/en/latest/miniconda.html) to download python 3.8 or higher and then

```
pip install -r requirements.txt
```

## 🎈 Usage <a name="usage"></a>

To run the website, navigate to main folder of the project
```
python app.py
```

The server will be at "localhost:5000".

Goto "localhost:5000" and after entering the comment click on classify to predict it's toxicity values.

<hr>

## 🚀 Deployment <a name = "deployment"></a>

The model has been deployed on an EC2 instance on AWS. The IP has been made publicly accesible. Below is the link to the AWS webapp project portal:

Link: http://ec2-18-117-78-151.us-east-2.compute.amazonaws.com:8080/

## 🌟 Support <a name="support">

Please hit the ⭐button if you like this project. 😄

# Thank you!