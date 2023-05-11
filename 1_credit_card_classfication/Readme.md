# Default of credit card clients

### Table of Contents

1. [Chapter 1 - Project Overview](#Project-Overview)
2. [Chapter 2 - Data Science Steps](#Data-Science-Steps)
3. [Chapter 3 - Step 1: Problem Definition](#Step-1-Problem-Definition)
4. [Chapter 4 - Step 2: Data Gathering](#Step-2-Data-Gathering)
5. [Chapter 5 - Step 3: Data Preparation](#Step-3-Data-Preperation)
6. [Chapter 6 - Step 4: Explanatory Data Analysis (EDA)](#Step-4-Explanatory-Data-Analysis-(EDA))
7. [Chapter 7 - Step 5: Data Modelling](#Step-5-Data-Modelling)
8. [Chapter 8 - Evaluate Model Performance](#Step-6-Predicting-Test-dataset)
9. [Chapter 9 - Conclusion](#Conclusion)
10. [References](#Reference)

### Project Overview

This Project aimed at the case of customers default payments in Taiwan and compares the predictive accuracy of probability of default. From the perspective of risk management, the result of predictive accuracy of the estimated probability of default will be more valuable than the binary result of classification - credible or not credible clients. 

I worked with the project in Default of credit card clients. I focused on predicting the Default list of customer. I followed a process of problem definition, gathering data, preparing data, explanatory data analysis, coming up with a data model, validating the model, and optimizing the model further.

Let's take a look at the steps


### Data Science Steps

1. Problem Definition: What factors determined whether someone survived a disaster? Using passenger data, we were able to identify certain groups of people who were more likely to survive.
2. Data Gathering: Kaggle provided the input data on their website.
3. Data Preparation: I prepared the data by analyzing data points that were missing or outliers.
4. EDA (Exploratory Data Analysis): If you input garbage data into a system, you'll get garbage output. Therefore, it is important to use descriptive and graphical statistics to look for patterns, correlations and comparisons in the dataset. In this step, I analyzed the data to make sure it was understandable.
5. Data Modeling: It is important to know when to select a model. If we choose the wrong model for a particular use case, all other steps become pointless.
6. Validate Model: After training the model, I checked its performance and looked for any issues with overfitting or underfitting.
7. Optimize Model: Using techniques like hyperparameter optimization, I worked on making the model better.



### Step 1 Problem Definition

This research aimed at the case of customers default payments in Taiwan and compares the predictive accuracy of probability of default among different models methods.



### Step 2 Data Gathering
The dataset can be found on found on [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

### Step 3 Data Preperation

The data was pre-processed, so I only focused on cleaning it up further.



#### 3.1 Import Libraries



```python
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

import pickle

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
    


```python
#Common Model Algorithms
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegressionCV

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import roc_auc_score

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
pd.set_option('display.max_rows', 30)
```

#### 3.2 Pre-view of the Data


The default payment next month variable is the outcome or dependent variable. The datatype is 1 if the customer default and 0 if they did not default. The rest of the variables are independent variables. Most variable names are self explanatory but a couple may be worth mentioning. The LIMIT_BAL represents Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.Pay_6 to Pay_0 is the History of past payment. We tracked the past monthly payment records (from April to September, 2005).similarly for BILL_AMT & PAY_AMT columns respectively.

![image.png](/imagesfile/tree/main/1_credit_card_classfication/a46ca2ee-ace7-493d-bc89-a96123207b49.png)

![image.png](attachment:29824355-38b5-4e02-a987-7af61b14a351.png)

![image.png](attachment:f813f3f5-b78b-4efb-8aae-72761f96817c.png)

Check for missing values

![image.png](attachment:a5728728-e35e-40fd-9f36-afb9f5072d41.png)

Ratio between Male and female (1 = male; 2 = female)

![image.png](attachment:f2eb6ec4-2b3d-4e6d-a4ae-50bd834eed25.png)

#### 3.3 Data Pre-processing:


##### 3.3.1 Rename Columns by month

PAY_0 to PAY_6 history of past payment. We tracked the past monthly payment records (from April to September, 2005), similarly for BILL_AMT & PAY_AMT columns respectively.

To change the columns accordingly to respective month to get clear idea of what the features does.  
- PAY_0,	PAY_2,	PAY_3,	PAY_4,	PAY_5,	PAY_6 changed to PAY_September,	PAY_August,	PAY_July,	PAY_June, PAY_May, PAY_April
- BILL_AMT1,	BILL_AMT2,	BILL_AMT3,	BILL_AMT4,	BILL_AMT5,	BILL_AMT6, to BILL_AMT_September,	BILL_AMT_August,	BILL_AMT_July,	BILL_AMT_June,	BILL_AMT_May,	BILL_AMT_April
- PAY_AMT1,	PAY_AMT2,	PAY_AMT3,	PAY_AMT4,	PAY_AMT5,	PAY_AMT6, to PAY_AMT_September,	PAY_AMT_August,	PAY_AMT_July,	PAY_AMT_June,	PAY_AMT_May,	PAY_AMT_April
- Rename the Target column to avoid space/column length


![image.png](attachment:900ee1d7-27a6-47b9-908a-5d56cd3ab489.png)

![image.png](attachment:d144beb7-6c93-4b2d-9457-36f935414363.png)

Rounding the value to four categories to understand easily

![image.png](attachment:01b08bff-6980-4418-ae1a-0b24229b2e85.png)

### Step 4 Explanatory Data Analysis

- Below graph shows Male mosty get defaulted than Female
- highschool gets defaulted than other categories 
- Married person get defaulted than single 


![image.png](attachment:659879ca-349e-407c-9eb7-106987730533.png)

Male at all age bins get defaulted than Female
as we seen above married male gets default than married female
![image.png](attachment:4b7483a7-54fc-4e04-9293-169c068718c9.png)

As we seen below low limit score mostly to get default  

![image.png](attachment:899a38be-d393-4bdd-9d4e-9001ed0a4217.png)


Pay_september status > 2 mostly gets defaulted 

![image.png](attachment:64e3bca0-d610-4c5c-8d4c-6bd430d6ebf3.png)

Predicting the Good feature for our model using ANOVA for All features vs Target. Pay_status features will impact more than others lets compare later with feature importance 
 

![image.png](attachment:b6cbf69b-6a6c-430b-a678-2022588d5901.png)

Top 25% feature are listed below to get good output

![image.png](attachment:d85e6a1a-a66b-4ef0-aca9-f9c36c374d13.png)

Heatmap of the entire dataset:

![image.png](attachment:e75ceefe-719c-4fda-b278-6f2d1ee35c2d.png)

### Step 5 Data Modelling

I will use a supervised learning classification algorithm for predicting the binary ourcome (Default or not). Here are some resources I found helpful:

Below Machine Learning Classification Algorithms used to predict output:
- [Logistic Regression](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors)
- [XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html)

#### 5.1 Evaluate Model Performance

AUC score for Logistic Regression algorithm:

![image.png](attachment:98b0a1e8-1b5a-48e1-8efe-76ed9dd0c719.png)

The confusion matrix:

![image.png](attachment:503e517b-36a7-4fc7-9457-8a810a565f07.png)

![image.png](attachment:f62349af-1e5d-4aaa-8fea-aad3ad5d0f35.png)

A poor Precision recall and ROC curve

![image.png](attachment:ad5e908b-035b-495e-b301-36a2976381c7.png)

![image.png](attachment:d3aca925-2820-4212-8fce-e1bdd23b4623.png)

#### 5.2 Tune Model with Hyper-Parameters for XGBClassifier

Best Model Performance with Cross-Validation for XGBClassifier is 0.74 AUC

![image.png](attachment:a8d60118-6e48-4e15-8878-4b75970a4e0a.png)

#### 5.3 Explaining Model Predictions with SHAP Values

The SHAP value indicates the additive
contribution of each feature value to the prediction for a sample. Negative SHAP values indicate a feature moving the prediction closer to 0, and positive SHAP values indicate closer to 1.
These represent the values of the features for each sample, with red meaning a higher value and blue lower. So, for example, we can see in the Pay_september column the highest SHAP values come
from high feature values (red dots). Pay_sept > 2 mostly get default list  

![image.png](attachment:8d9dc1e8-ddac-47b9-9e86-6f4658debb65.png)

#comparing ANOVA F-value with Top 25% features & SHAP summart plot

- Statistical test and shap plot shows same result 
- Important features are "PAY_September", "LIMIT_BAL", "PAY_August", "BILL_AMT_September"

![image.png](attachment:f8e6514a-8ba1-4a6d-8a9a-25bce7cd4b78.png)


![image.png](attachment:2722a64d-be1f-46bb-bb5e-5c296f03949a.png)

#### 5.4 Tune Model with Feature Selection

As we seen in above graph value of Pay_sept & Pay_Aug gets higher mostly likely to get defaulted. Lower value for limit_bal move towards default list

![image.png](attachment:fba121b2-22d2-40ce-976c-d1a7f57f15e8.png)

![image.png](attachment:2a6e8c59-d3ba-4190-bf71-928f79e25f82.png)

Lower limit_bal with Pay_sept >0 will mostly be default


![image.png](attachment:db6d69bf-8f7a-4a10-b4a8-ceb245702c5b.png)

Below graph clearly can be seen Limit_bal <200000 mostly fall in Pay_sept[1-8] will get defaulted

![image.png](attachment:4e4d3dfa-0eb1-4705-a649-dfdace4c8142.png)

Similar below graph shows same as Pay_sept & Pay_Aug(past month)

![image.png](attachment:6ea6b4a4-0314-46e3-b003-75a2c8a69aa9.png)

### Step 6 Predicting Test dataset

#### 6.1 Distribution of Predicted Probability and Decile Chart

According to the model, most borrowers have a 0-20% chance of default.

![image.png](attachment:4620c5a1-ba70-4b2d-afe0-1f64da0cbbcf.png)

Default risk increases with each decile,where the riskiest 10% of borrowers have a default rate close to 70%, but the least risky are below 10%. When a model is able to effectively distinguish groups of
borrowers with consistently increasing default risk, the model is said to slope the population being examined.

Notice also that the default rate is relatively flat across the lowest 5 to 7 deciles, likely because these observations are mostly clustered in the range [0, 0.2] of predicted risk

![image.png](attachment:669ec44c-716b-48e9-a2e8-0c3e6dd36cbf.png)

#### 6.2 Calibration of Predicted Probabilities

Below value started to increase our model becoming less calibrated 

![image.png](attachment:359e482a-219c-4c17-aacc-6974afc4503f.png)

Below graph shows that model-predicted probabilities are very close to the true default rates, so the model appears to be well calibrated.

![image.png](attachment:a8a1765b-4299-4c7f-b745-8fb97ba88b0a.png)

#### 6.3 Financial Analysis

The plot indicates that the choice of threshold is important. While it will be
possible to create net savings at many different values of the threshold, it
looks like the highest net savings will be generated by setting the threshold
somewhere in the range of about 0.25 to 0.5.

max_savings we get $13,836,743.7 at Threshold set to 0.32

![image.png](attachment:9c1276b9-96f3-439a-a35a-3bccdb0dff56.png)

- Cost of all defaults if there were no counseling program is $61,786,034

- Total net_savings from our model in given test dataset is $13,836,743.7

- Net savings per account at the optimal threshold relative to the whole test is $2,306.12


Net savings per account against the cost of counseling per account for each threshold

![image.png](attachment:6d2355a1-b278-4e27-8b8f-ca4642954d4c.png)

Fraction of accounts predicted as positive (this is called the "flag rate") at each threshold. Set the optimal threshold of 0.32, only about 20% of accounts will be flagged for counseling.

![image.png](attachment:cf6a7613-6a28-457c-8ca3-3b99db8ae491.png)

precision and recall separately on the y-axis against threshold on the x-axis. It shows as same optimal threshold will be 0.32

![image.png](attachment:dc8e415b-b163-4549-b19f-688d9254542c.png)

### Conclusion

Based on the exploratory data analysis, we discover that human characteristics are not the most important predictors of default, the payment status of the most 2 months and credit limit

From the modeling, we are able to classify default risk with accessible customer data and find a decent model. Using a Logistic Regression classifier, we can predict with 0.72 AUC, whether a customer is likely to default next month. Using a XG BOOST classifier, we can predict with 0.74 AUC, whether a customer is likely to default next month.

If the threshold is set at 0.32, then the model in the given test dataset would save $13,836,743.7

### Reference

I would like to thank the following resources and developers:
- [Data Science Projects with Python second edition by Stephen Klosterman](https://www.amazon.com/Data-Science-Projects-Python-approach/dp/1800564481)


```python

```


```python

```


```python

```
