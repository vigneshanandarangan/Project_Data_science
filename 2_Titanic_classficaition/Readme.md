## Titanic - Machine Learning from Disaster

I focused on predicting the survival data for passengers after a disaster.




## Table of Contents

1. [Chapter 1 - Project Overview](#Project-Overview)
2. [Chapter 2 - Data Science Steps](#Data-Science-Steps)
3. [Chapter 3 - Step 1: Problem Definition](#Step-1-Problem-Definition)
4. [Chapter 4 - Step 2: Data Gathering](#Step-2-Data-Gathering)
5. [Chapter 5 - Step 3: Data Preparation](#Step-3-Data-Preperation)
6. [Chapter 6 - Step 4: Explanatory Data Analysis (EDA)](#Step-4-Explanatory-Data-Analysis)
7. [Chapter 7 - Step 5: Data Modelling](#Step-5-Data-Modelling)
8. [Chapter 8 - Evaluate Model Performance](#Step-6-Evaluate-Model-Performance)
9. [Chapter 9 - Model Performance with Cross-Validation (CV)](#Step-7-Model-Performance-with-Cross-Validation)
10. [Chapter 10 - Tune Model with Hyper-Parameters](#Step-8-Tune-Model-with-Hyper-Parameters-with-voting-classifier)
12. [Chapter 11 - Explain Model  with SHAP](#Step-9-Explaining-Model-Predictions-with-SHAP-Values)
13. [Chapter 13 - Conclusion](#Conclusion)
14. [References](#References)



### Project Overview

In this project, I chose a very popular example of classification. It is either 0 or 1. It either happened or it didn't happen. For example, cancer is positive or not, the production part is working or not working, tomorrow it will rain or not, etc.

I worked with the project in Kaggle's Getting Started Competition, Titanic: Machine Learning from Disaster. I focused on predicting the survival data for passengers. I followed a process of problem definition, gathering data, preparing data, explanatory data analysis, coming up with a data model, validating the model, and optimizing the model further.

Let's take a look at the steps:

### Data Science Steps

1. Problem Definition: What factors determined whether someone survived a disaster? Using passenger data, we were able to identify certain groups of people who were more likely to survive.
2. Data Gathering: Kaggle provided the input data on their website.
3. Data Preparation: I prepared the data by analyzing data points that were missing or outliers.
4. EDA (Exploratory Data Analysis): If you input garbage data into a system, you'll get garbage output. Therefore, it is important to use descriptive and graphical statistics to look for patterns, correlations and comparisons in the dataset. In this step, I analyzed the data to make sure it was understandable.
5. Data Modeling: It is important to know when to select a model. If we choose the wrong model for a particular use case, all other steps become pointless.
6. Validate Model: After training the model, I checked its performance and looked for any issues with overfitting or underfitting.
7. Optimize Model: Using techniques like hyperparameter optimization, I worked on making the model better.

### Step 1 Problem Definition

The goal of this project is to predict the survival outcomes of passengers on the Titanic.

Project Summary from Kaggle: The sinking of the Titanic is one of the most famous maritime disasters in history. On April 15, 1912, the RMS Titanic sank after colliding with an iceberg. This was considered to be an unsinkable ship, but nonetheless it went down due to the accident. Unfortunately, there weren't enough lifeboats for everyone on the ship, resulting in the death of 1502 people out of 2224 passengers and crew.

Some groups of people seemed to be more likely to survive than others, although luck was involved. In this challenge, they want us to create a predictive model that can identify who is more likely to survive based on data about passengers (name, age, gender, social class, etc).

### Step 2 Data Gathering
The dataset is also given to us on a golden plater with test and train data at [Kaggle's Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)



### Step 3 Data Preperation
The data was pre-processed by Kaggle, so I only focused on cleaning it up further.



#### 3.1 Import Libraries



```python
# #load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import random
import time


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)

```

    Python version: 3.9.13 (main, Aug 25 2022, 23:51:50) [MSC v.1916 64 bit (AMD64)]
    pandas version: 1.4.4
    matplotlib version: 3.5.2
    NumPy version: 1.21.5
    SciPy version: 1.9.1
    IPython version: 7.31.1
    scikit-learn version: 1.0.2
    -------------------------
    

#### 3.2 Common Model Algorithms



```python
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.width', 75)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 12)
```

#### 3.3 Pre-view of the Data


The Survived variable is the outcome or dependent variable. The datatype is 1 if the person survived and 0 if they did not survive. The rest of the variables are independent variables. Most variable names are self explanatory but a couple may be worth mentioning. The SibSp represents the number of siblings / spouses aboard the Titanic and Parch represents the number of parents / children aboard the Titanic.



![image.png](attachment:3e3df65e-c777-42aa-bb88-dc5cb5a00037.png)

![image.png](attachment:8fe0580f-1d6e-42c2-a1d7-d90a180c1ba0.png)

#### 3.4 Data Pre-processing:


I cleaned the data by identifying and removing abnormal values and outliers, filled in missing data where appropriate, worked on improving the features, and performed data conversion. I used Label encoder to convert objects to categories.

Divided the data into 75/25 format. 75 is training and 25 is test.




### Step 4 Explanatory Data Analysis


After cleaning and organizing the data, it is important to explore it in order to find any insights. I used EDA to visualize the data I am working with, in order to better understand its properties and statistics.



There are many outliers present in Continuous variables

![image.png](attachment:1f9e8aea-3991-4e43-a90a-40e81db5d430.png)

- Lower Fare mostly Dead
- Travelling alone mostly dead

![image.png](attachment:95dc0798-680e-4635-a9fc-8b5a98675682.png)



Looking at individual features by survival:

![image.png](attachment:aed7c2f2-eddd-47bb-a514-6e3c8c11208f.png)


![image.png](attachment:b2c69104-a335-43f0-a080-636adcf2686c.png)

Graph distribution of qualitative data: Pclass

![image.png](attachment:a5376ef1-066d-4dc2-be37-7c4981e0ae13.png)

we know Gender plays huge role in survival, now let's compare sex and a 2nd feature

![image.png](attachment:936351d7-2e03-4538-b926-33e80dac9d51.png)


- how does family size factor with sex & survival compare

- how does class factor with sex & survival compare

![image.png](attachment:864a6781-23cd-453c-89ed-b843c647d6d8.png)

how does embark port factor with class, sex, and survival compare

![image.png](attachment:99e78fe1-71db-4cef-93b2-b26eaa71f6cc.png)


#plot distributions of age of passengers who survived or did not survive

![image.png](attachment:944636e4-88df-4f3f-9115-42c67387b754.png)

Histogram comparison of sex, class, and age by survival:

![image.png](attachment:0c4d8968-228f-4657-8926-11edd3583895.png)

Pairplot to see the entire dataset:

![image.png](attachment:7c3b6f51-e14a-4df4-a954-805ea1466fd6.png)

Heatmap of the entire dataset:

![image.png](attachment:d1b0169d-543a-43b2-bf02-2a0a2431dfa9.png)

### Step 5 Data Modelling

I will use a supervised learning classification algorithm for predicting the binary ourcome (survived or not). Here are some resources I found helpful:

Some Machine Learning Classification Algorithms:

- [Ensemble Methods](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)
- [Generalized Linear Models (GLM)](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
- [Naive Bayes](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes)
- [Nearest Neighbors](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors)
- [Support Vector Machines (SVM)](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)
- [Decision Trees](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree)
- [Discriminant Analysis](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.discriminant_analysis)
- [XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html)

#### Which Machine Learning Algorithm (MLA) to choose ?

Deciding on which model to use is never a straight answer. In practice, best approach is to work on different algorithms and then compare their performace. Below I summarized the models I worked with, and their performances:


![image.png](attachment:08dfcf1b-f062-4bb4-8ec2-b15013b62e10.png)

Then let's see the barplot showing the accuracy score Train & test dataset:



![image.png](attachment:f9e6114f-acc8-413d-a346-faec7ca747b2.png)

### Step 6 Evaluate Model Performance

After some data pre-processing, analysis, and machine learning algorithms (MLA), I was able to predict passenger survival with ~81% accuracy. 

![image.png](attachment:e6d2bc0c-c748-4275-b15d-1723f25f9404.png)


The confusion matrix without normalization:

![image.png](attachment:a1283ef3-9649-4d64-8ac6-397f81c2e8f6.png)

Confusion matrix with normalization:

![image.png](attachment:35ac3abf-406c-4d53-ae8e-5ac4d75ea05a.png)

### Step 7 Model Performance with Cross-Validation 

In this section, I worked on cross validation (CV). I leveraged the [sklearn cross_validate function](https://scikit-learn.org/stable/modules/cross_validation.html#multimetric-cross-validation). Some advantages of sklearn cross_validate function are:

- It allows specifying multiple metrics for evaluation.
- It returns a dict containing fit-times, score-times in addition to the test score.

By using CV I was also automatically able to split and score the model multiple times, to get an idea of how well it will perform on unseen data.

### Step 8 Tune Model with Hyper-Parameters with voting classifier

I worked on hyper-parameter optimization to see how various hyper-parameter settings will change the model accuracy.

Decision trees are simple to understand usually. They can also be visualized. Data prep is quite easy compared to other methods. They can handle both numeric and categorical data. We can validate a model using tests.

However, decision trees do not generalize data well, they do have a tendency to memorize (overfitting). Pruning can be used to overcome this issue. Small variations may impact the decision trees hugely. They can be biased if some classes dominate.

why choose one model, when you can pick them all with voting classifier



voting classifier before optimization:

![image.png](attachment:9c16a974-1a49-4a10-96d0-3d386a7a9ff9.png)

voting classifier after optimization:

![image.png](attachment:d26c5e00-ce3b-40c1-9a2b-bfd04d5f2b47.png)

### Step 9 Explaining Model Predictions with SHAP Values

 SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model.
 
 The most important features used by XGBClassifier are SEX and Pclass, which we have already seen in EDA.

Note: Since voting classifiers cannot use SHAP, we used the XGB model to explain the model's predictions.


 ![image.png](attachment:2032dab5-1ddf-490f-8b0c-1c8e0ea299a6.png)


Sample value from train set

Males in third class, and that the predicate is did not survive the sinking of the Titanic

![image.png](attachment:dd929ba4-5f77-41b4-a43e-081d8eb78a53.png)

Females in first class,and that the predicate is survived the disaster

![image.png](attachment:45975635-796d-4e20-b627-f4226c00a999.png)

### Conclusion

The model achieved an accuracy of about 78% when predicting the outcomes of unseen data.

![image.png](attachment:30adada4-949c-474e-9eb5-e40a243d2e7d.png)

### References
I would like to thank the following resources and developers:

- [Kaggle: Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview) - Kaggle project overview.




```python

```
