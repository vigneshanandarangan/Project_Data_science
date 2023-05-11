## Wine Quality Analysis

### Table of Contents


1. [Chapter 1 - Project Overview](#Project-Overview)
2. [Chapter 2 - Data Science Steps](#Data-Science-Steps)
3. [Chapter 3 - Step 1: Problem Definition](#Step-1-Problem-Definition)
4. [Chapter 4 - Step 2: Data Gathering](#Step-2-Data-Gathering)
5. [Chapter 5 - Step 3: Data Preparation](#Step-3-Data-Preperation)
6. [Chapter 6 - Step 4: Explanatory Data Analysis (EDA)](#Step-4-Explanatory-Data-Analysis)
7. [Chapter 7 - Step 5: Data Modelling](#Step-5-Data-Modelling)
8. [Chapter 8 - Conclusion](#Conclusion)



### Project Overview

Wine is an alcoholic beverage made from fermented grapes. Yeast consumes the sugar in the grapes and converts it to ethanol, carbon dioxide, and heat. It is a pleasant tasting alcoholic beverage, loved cellebrated . It will definitely be interesting to analyze the physicochemical attributes of wine and understand their relationships and significance with wine quality and types classifications. To do this, We will proceed according to the standard Machine Learning and data mining workflow models like the CRISP-DM model, mainly for:

Predict if each wine sample is a red or white wine.
Predict the quality of each wine sample, which can be low, medium, or high.
The dataset are related to red and white variants of the "Vinho Verde" wine. Vinho verde is a unique product from the Minho (northwest) region of Portugal. Medium in alcohol, is it particularly appreciated due to its freshness (specially in the summer). This dataset is public available for research purposes only, for more information, read Cortez et al., 2009. . Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

### Data Science Steps


1. Problem Definition: What factors determined whether someone survived a disaster? Using passenger data, we were able to identify certain groups of people who were more likely to survive.
2. Data Gathering: Kaggle provided the input data on their website.
3. Data Preparation: I prepared the data by analyzing data points that were missing or outliers.
4. EDA (Exploratory Data Analysis): If you input garbage data into a system, you'll get garbage output. Therefore, it is important to use descriptive and graphical statistics to look for patterns, correlations and comparisons in the dataset. In this step, I analyzed the data to make sure it was understandable.
5. Data Modeling: It is important to know when to select a model. If we choose the wrong model for a particular use case, all other steps become pointless.
6. Validate Model: After training the model, I checked its performance and looked for any issues with overfitting or underfitting.


### Step 1 Problem Definition
Here we will predict the quality of wine on the basis of given features. We use the wine quality dataset available on Internet for free. This dataset has the fundamental features which are responsible for affecting the quality of the wine. By the use of several Machine learning models, we will predict the quality of the wine.

### Step 2 Data Gathering


The dataset can be found on found on [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)

### Step 3 Data Preperation

The data was pre-processed, so I only focused on cleaning it up further.

#### 3.1 Import Libraries



```python
import pandas  as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew
from scipy.stats import pearsonr
sns.set(rc={'figure.figsize': (14, 8)})
import warnings
warnings.filterwarnings('ignore')
```


```python
#Common Model Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score 
```

Red Wine

![image.png](/images/3_Wine_EDA/b2f00e2c-b6e8-46cd-8940-4350b78843c1.png)

![image.png](/images/3_Wine_EDA/671a6471-9bc5-4169-be0b-a6cca40e2ef8.png)

White Wine

![image.png](/images/3_Wine_EDA/3762bf22-d0f0-466d-a9af-4f023f901aea.png)

![image.png](/images/3_Wine_EDA/9652406b-ada9-48a5-af54-819a0e7d4141.png)

#### 3.2 Nulls Check and Cleaning

Red & White wines have zero null values


![image.png](/images/3_Wine_EDA/68d33f96-4f95-487f-90d3-5bc3357f2ca6.png)

### Step 4 Explanatory Data Analysis

majority of the red wine belongs to the group with quality labels 5 and 6

![image.png](/images/3_Wine_EDA/dec1af30-52d3-462c-8ac6-ea26d1973696.png)

- Positive correlation between fixed acidity and density.
- Negative correlation between acidity and pH.
- Negative correlation between alcohol percentage and density.


![image.png](/images/3_Wine_EDA/3cea0bac-10a1-4fe0-9736-49b34a09a757.png)

The quality column has a positive correlation with alcohol, sulfates, residual sugar, citric acid, and fixed acidity.
- Alcohol is positively correlated with the quality of the red wine.
- Alcohol has a weak positive correlation with the pH value.
- Citric acid and density have a strong positive correlation with fixed acidity.
- pH has a negative correlation with density, fixed acidity, citric acid, and sulfates.

![image.png](/images/3_Wine_EDA/9375605b-ef60-415d-84bc-d9217a9dff16.png)

Alcohol distribution is positively skewed with the quality  of the red wine.

![image.png](/images/3_Wine_EDA/79470f92-28e9-4bea-a739-253e549f96bc.png)

The output verifies that alcohol is positively skewed. That gives deeper insight into the alcohol column.

![image.png](/images/3_Wine_EDA/c2fbc209-4724-4197-838c-d5eea34abdd1.png)



Boxplot helps to display the outlier here quality 5 has more outlier than others

![image.png](/images/3_Wine_EDA/9c6aa54d-5b07-4c91-b177-a7d8ea99419e.png)

Alcohol is weakly positively related to the pH values.

![image.png](/images/3_Wine_EDA/8804e4d3-19b3-4290-858e-1f07e7312351.png)


Combining Red & White wine dataset and labled the quality as "Low", "Medium & "High".

![image.png](/images/3_Wine_EDA/77c9afbd-7c9f-4f71-b01b-2bd9a0142edc.png)


We have grouped the dataset into three distinct groups: low-quality wine, medium-quality wine, and high-quality wine. Each group shows three different attributes: alcohol, density, and pH value


![image.png](/images/3_Wine_EDA/16a7d5c2-6a5a-4331-a39b-9adf2e4be5ad.png)

#### 4.1 Univariate analysis

Visualize the numeric data and their distribution is by using a histogram.

![image.png](/images/3_Wine_EDA/d827b9a9-1043-4c54-abaf-de87a3c5694a.png)

#### 4.2 Multivariate analysis

Correlation between the features, Alcohol & density have high correlation(-0.69) than others.


![image.png](/images/3_Wine_EDA/773da902-7ac3-4a24-a16a-5af3b6bde6b8.png)

The count plot shows the frequency distributions of different categories of wine, namely 3, 4, 5, 6, 7, 8, and 9.

![image.png](/images/3_Wine_EDA/f39f8975-3e4f-47ef-86ed-96f77b6da50a.png)

### Step 5 Data Modelling

The random forest, decision tree, gradient boosting classifier, and the Gaussian Naive Bayes classifier all achieved an accuracy of 100%.


![image.png](/images/3_Wine_EDA/0c39d0f7-1550-4f31-bd1a-0593d4a40f29.png)

![image.png](/images/3_Wine_EDA/bc584f0d-c22c-4341-b9b1-67d49c272d1e.png)

### Conclusion

We conclude that it is possible to predict the quality and type of wine from its physical and chemical attributes. While the prediction of type is only of educational value, the prediction of quality has some practical applications, such as:



- The result of the model, its interpretation and all the evaluations of EDA provide methods and rules of decision that can help winemakers to look for wines of better qualities.
- Assisting consumers in choosing wines that they are likely to enjoy.



```python

```
