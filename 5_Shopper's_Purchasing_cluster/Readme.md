## Analyzing the Online Shopper's Purchasing Intention

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

Consumer shopping on the internet is growing year by year. However, the conversion
rates have remained more or less the same. For example, most of us browse through
e-commerce websites such as Amazon, perhaps adding items to a wish list or even an
online shopping cart, only to end up buying nothing.

From this common truth comes the need for tools and solutions that can customize
promotions and advertisements for online shoppers and improve this conversion. We will be analyzing various factors that affect a purchaser's decision.



### Data Science Steps


1. Problem Definition: What factors determined whether someone survived a disaster? Using passenger data, we were able to identify certain groups of people who were more likely to survive.
2. Data Gathering: Kaggle provided the input data on their website.
3. Data Preparation: I prepared the data by analyzing data points that were missing or outliers.
4. EDA (Exploratory Data Analysis): If you input garbage data into a system, you'll get garbage output. Therefore, it is important to use descriptive and graphical statistics to look for patterns, correlations and comparisons in the dataset. In this step, I analyzed the data to make sure it was understandable.
5. Data Modeling: It is important to know when to select a model. If we choose the wrong model for a particular use case, all other steps become pointless.
6. Validate Model: After training the model, I checked its performance and looked for any issues with overfitting or underfitting.

### Step 1 Problem Definition
Analyzing the Online Shopper's Purchasing and also be able to implement clustering and make recommendations based on the predictions. These recommendations will help you gain actionable insights and make
effective decisions

### Step 2 Data Gathering

data from the Online Shoppers Purchasing
Intention Dataset, which can be obtained from the [UCI repository.](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)

### Step 3 Data Preperation

The data was pre-processed, so I only focused on cleaning it up further.

#### 3.1 Import Libraries



```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

```

Sample from a dataset 

![image.png](attachment:bb7f2bbc-ea69-4835-8c3e-2475c27f06dc.png)

Data has no missing values

![image.png](attachment:5165fb76-8dcc-40d8-ac3c-bdf7ebe64e28.png)

![image.png](attachment:fc2011f4-b96c-4495-ae11-78cb78811274.png)

### Step 4 Explanatory Data Analysis

#### 4.1 Univariate Analysis

plot distributions of numerical features

![image.png](attachment:7c73cc88-1c77-4a42-8a26-e5f5acc83223.png)

Analyzing each plot separately

It can be seen that the number of False values is greater than the number of True values.


![image.png](attachment:a1fc6e52-cd55-476b-8cc5-57cee838caa7.png)

![image.png](attachment:7ef576aa-53af-495f-93cf-8293137f8b4e.png)

![image.png](attachment:c00c904b-dbcd-4e93-929d-67464185a38e.png)

The number of returning customers is higher than that of new visitors. This is good news as it means we have been successful in attracting customers back to our website.

![image.png](attachment:927c4a8f-b540-47c8-8bab-e4556a677a83.png)




![image.png](attachment:81c11528-12c3-48a2-985b-54ab1173010f.png)

From the preceding information, we can see that sources 2, 1, 3, and 4 account for the majority of our web traffic.

![image.png](attachment:de77df20-3f6c-48f0-a0a3-34627a287d97.png)




![image.png](attachment:936f4269-57b7-4089-a9d2-da8b2807f01b.png)

From the count of the False subcategory, we can see that more visitors visit during weekdays than weekend days.

![image.png](attachment:e8c431dc-28d3-4c49-90d5-13fad7ea2c2c.png)




![image.png](attachment:480f49d3-5781-4804-a3e0-f02ce8c177bb.png)

From the preceding data, we can see that regions 1 and 3 account for 50% of online sessions; thus, we can infer that regions 1 and 3 
are where most potential consumers reside. 
With this information, we can target our marketing campaigns better.

![image.png](attachment:680cb113-c447-459d-b07f-566f2c47d841.png)





![image.png](attachment:2700dedd-f2b7-4c80-a769-dfdfc2311a30.png)

From the preceding graph, we can see that browser type 2 contributes the most
to the website traffic.

![image.png](attachment:cda6e993-57f2-42c1-a519-12cfd55fd4a6.png)




![image.png](attachment:ea05b787-ddaa-4152-a39c-94a36406a220.png)

If we know which OS type is the most predominant, we can ask the tech team to configure the website for that particular OS and take the
necessary actions, such as explicitly defining CSS for that particular OS.

![image.png](attachment:22a9258b-4365-482a-95e4-ac33676ded06.png)


We can see from the preceding plot that users tend to visit page 0 the most often.

![image.png](attachment:8a30df09-7647-4e8a-a53a-cdbd87d20507.png)




#### 4.2 Bivariate Analysis

As you can see, more revenue conversions happen for returning customers than new customers. This clearly implies that we need to find
ways to incentivize new customers to make a transaction with us

![image.png](attachment:a02bc5f4-1e1d-4c16-bab0-0d7a7966af1e.png)



From the preceding plot, we can see that more revenue conversion happens for web traffic generated from source 2. Even though source
13 generated a considerable amount of web traffic, conversion is very low compared to others

![image.png](attachment:80561222-a77a-4481-b672-cc5edb03e456.png)


From the preceding plot, we can see that region 1 accounts for most sales, and region 3 the second most. With this information, we
can plan our marketing and supply chain activities in a better way. For example, we might propose building a warehouse
specifically catering to the needs of region 1 to increase delivery rates and ensure that products in the highest demand are always
well stocked.

![image.png](attachment:238f30ad-8389-41c2-9ef7-256c111ce2c6.png)


Website visitors may be high in May, but we can observe from the preceding bar plot
that a greater number of purchases were made in the month of November.

![image.png](attachment:5357a716-02d5-4d22-9ce2-6dff4e1a319f.png)


From the preceding plot, we can infer that administrative-related pageviews and the administrative-related pageview duration are
positively correlated. When there is an increase in the number of administrative pageviews, the administrative pageview duration also
increases.



![image.png](attachment:bb264485-13ae-444e-b095-348b4b65d68e.png)


From the preceding plot, we can conclude the following:
- Information page views and information pageview duration are positively correlated. With an increase in the number of information pageviews, the information pageview duration also increases.
- Customers who have made online purchases visited fewer numbers of informational pages. This implies that informational pageviews don't have much effect on revenue generation.

![image.png](attachment:ebc985c1-d16b-4ac9-9759-81c4d7ddf086.png)


### Step 5 Data Modelling

Here we are using the Clustering technique for an unsupervised model

Clustering is an unsupervised learning technique in which you group categorically
similar data points into batches, called clusters. Here, we will be focusing on the
k-means clustering method.

K-means clustering is a clustering algorithm based on iterations where similar data
points are grouped into a cluster based on their closeness to the cluster centroid. This
means that the model runs iteratively to find the cluster centroid.

The optimum number of clusters for a dataset is found by using the elbow method.

#### 5.1 Performing K-means Clustering for Administrative Duration versus Bounce Rate


![image.png](attachment:7a5d8537-9632-4d14-ad33-67915917cd7f.png)

From the preceding elbow graph, we can see that k=2 is the optimum value for clustering. Now, let's run k-means clustering with k=2:

![image.png](attachment:9c79296b-df92-4a1e-93fc-751a17fb3127.png)

From the preceding graph, we can infer that the uninterested customer spends much less time in administrative pages compared
with target customers

#### 5.2 Performing K-means Clustering for Administrative Duration versus Exit Rate

![image.png](attachment:ee8b932e-4423-4959-99a7-2a9f45e8349e.png)

![image.png](attachment:5ae208ee-0fde-4206-8dd0-db7cfa7394f8.png)

From the preceding graph, we can infer that the uninterested customer exit the page higher than target customers

### Conclusion

From all the analysis we've performed in this chapter, we can conclude the following:

- The conversion rates of new visitors are high compared to those of returning customers.
- While the number of returning customers to the website is high, the conversion rate is low compared to that of new customers.

These factors will largely influence the next plan of action and open new avenues for
more research and new business strategies and plans.


```python

```
