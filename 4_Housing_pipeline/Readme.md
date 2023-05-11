## California Housing Prices


### Table of Contents
1. [Chapter 1 - Project Overview](#Project-Overview)
2. [Chapter 2 - Step 1: Problem Definition]()
3. [Chapter 3 - Step 2: Data Gathering](#Step-2-Data-Gathering)
4. [Chapter 4 - Step 3: Data Preparation](#Step-3-Data-Preperation)
5. [Chapter 5 - Step 4: Data Modelling](#Step-4-Data-Modelling)
6. [Chapter 6 - Conclusion](#Conclusion)

### Project Overview

#### Objectives
    
- There are a large number of factors that can affect the value of a house property (eg. location, size, condition, time), these factors can change quite substantially from one property to another
- The housing market itself is quite a volatile industry, and is quite dependent on demand and supply fluctuations, not to even mention economic factors such as interest rates & inflation, so its quite a challenge to predict the price variation over time
- It's also quite challenging to predict housing prices due to the limited data that is available, most datasets contain a limited number of features related to each property, such is why feature engineering is quite important
- As a result, it is generally quite difficult to accurately predict property prices that take into account all the factors that influence it
- The California housing dataaset contains different house related attributes for properties located in California



#### STUDY AIM


- The aim is to model is to predict the median_house_value which is our target variable
- Overcome missing data with a basic SimpleImputer Median function 
- Understand how to turn a simple model into your own sklearn comparible class, our aim won't be to create the most perfect model
- This is a regression type problem i have tried to built pipeline and predict the output no visualization added


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
    


```python
#Common Model Algorithms
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.width', 75)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 30)
```

### Step 1 Problem Definition

We collected information on the variables using all the block groups in California from the 1990 Cens us. In this sample a block group on average includes 1425.5 individuals living in a geographically compact area. Naturally, the geographical area included varies inversely with the population density. We computed distances among the centroids of each block group as measured in latitude and longitude. We excluded all the block groups reporting zero entries for the independent and dependent variables. The final data contained 20,640 observations on 9 variables. The dependent variable is ln(median house value).

### Step 2 Data Gathering

We use pandas.read_csv() function to read the csv file. In the bracket, we put the file path along with a quotation mark, so that pandas will read the file into a data frame from that address. The file path can be either an URL or your local file address.

Because the data has first row has header, we can add an argument headers = 1 inside the read_csv() method, so that pandas will automatically set the first row as a header.
You can also assign the dataset to any variable you create.

The dataset can be downloaded from [here](https://www.kaggle.com/datasets/camnugent/california-housing-prices)


```python
housing = pd.read_csv('D:\\python_dir_Jupyter\\Project_resume\\Data_science_project\\4_Housing_pipeline\\housing.csv')
housing.shape
```




    (20640, 10)




```python
housing.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.00</td>
      <td>880.00</td>
      <td>129.00</td>
      <td>322.00</td>
      <td>126.00</td>
      <td>8.33</td>
      <td>452,600.00</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.00</td>
      <td>7,099.00</td>
      <td>1,106.00</td>
      <td>2,401.00</td>
      <td>1,138.00</td>
      <td>8.30</td>
      <td>358,500.00</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.00</td>
      <td>1,467.00</td>
      <td>190.00</td>
      <td>496.00</td>
      <td>177.00</td>
      <td>7.26</td>
      <td>352,100.00</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.00</td>
      <td>1,274.00</td>
      <td>235.00</td>
      <td>558.00</td>
      <td>219.00</td>
      <td>5.64</td>
      <td>341,300.00</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.00</td>
      <td>1,627.00</td>
      <td>280.00</td>
      <td>565.00</td>
      <td>259.00</td>
      <td>3.85</td>
      <td>342,200.00</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>



median_house_value is the output we should predict

### Step 3 Data Preperation

The data was pre-processed, so I only focused on cleaning it up further.



```python
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
```


```python
housing["income_cat"].value_counts()
```




    3    7236
    2    6581
    4    3639
    5    2362
    1     822
    Name: income_cat, dtype: int64



Splitting the dataset in equal ration of income_cat in train and test dataset 



```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```


```python
strat_train_set
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
      <th>income_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12655</th>
      <td>-121.46</td>
      <td>38.52</td>
      <td>29.00</td>
      <td>3,873.00</td>
      <td>797.00</td>
      <td>2,237.00</td>
      <td>706.00</td>
      <td>2.17</td>
      <td>72,100.00</td>
      <td>INLAND</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15502</th>
      <td>-117.23</td>
      <td>33.09</td>
      <td>7.00</td>
      <td>5,320.00</td>
      <td>855.00</td>
      <td>2,015.00</td>
      <td>768.00</td>
      <td>6.34</td>
      <td>279,600.00</td>
      <td>NEAR OCEAN</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>-119.04</td>
      <td>35.37</td>
      <td>44.00</td>
      <td>1,618.00</td>
      <td>310.00</td>
      <td>667.00</td>
      <td>300.00</td>
      <td>2.88</td>
      <td>82,700.00</td>
      <td>INLAND</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14053</th>
      <td>-117.13</td>
      <td>32.75</td>
      <td>24.00</td>
      <td>1,877.00</td>
      <td>519.00</td>
      <td>898.00</td>
      <td>483.00</td>
      <td>2.23</td>
      <td>112,500.00</td>
      <td>NEAR OCEAN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20496</th>
      <td>-118.70</td>
      <td>34.28</td>
      <td>27.00</td>
      <td>3,536.00</td>
      <td>646.00</td>
      <td>1,837.00</td>
      <td>580.00</td>
      <td>4.50</td>
      <td>238,300.00</td>
      <td>&lt;1H OCEAN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15174</th>
      <td>-117.07</td>
      <td>33.03</td>
      <td>14.00</td>
      <td>6,665.00</td>
      <td>1,231.00</td>
      <td>2,026.00</td>
      <td>1,001.00</td>
      <td>5.09</td>
      <td>268,500.00</td>
      <td>&lt;1H OCEAN</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12661</th>
      <td>-121.42</td>
      <td>38.51</td>
      <td>15.00</td>
      <td>7,901.00</td>
      <td>1,422.00</td>
      <td>4,769.00</td>
      <td>1,418.00</td>
      <td>2.81</td>
      <td>90,400.00</td>
      <td>INLAND</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19263</th>
      <td>-122.72</td>
      <td>38.44</td>
      <td>48.00</td>
      <td>707.00</td>
      <td>166.00</td>
      <td>458.00</td>
      <td>172.00</td>
      <td>3.18</td>
      <td>140,400.00</td>
      <td>&lt;1H OCEAN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19140</th>
      <td>-122.70</td>
      <td>38.31</td>
      <td>14.00</td>
      <td>3,155.00</td>
      <td>580.00</td>
      <td>1,208.00</td>
      <td>501.00</td>
      <td>4.20</td>
      <td>258,100.00</td>
      <td>&lt;1H OCEAN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19773</th>
      <td>-122.14</td>
      <td>39.97</td>
      <td>27.00</td>
      <td>1,079.00</td>
      <td>222.00</td>
      <td>625.00</td>
      <td>197.00</td>
      <td>3.13</td>
      <td>62,700.00</td>
      <td>INLAND</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>16512 rows × 11 columns</p>
</div>




```python
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
```




    3   0.35
    2   0.32
    4   0.18
    5   0.11
    1   0.04
    Name: income_cat, dtype: float64




```python
strat_train_set["income_cat"].value_counts() / len(strat_train_set)
```




    3   0.35
    2   0.32
    4   0.18
    5   0.11
    1   0.04
    Name: income_cat, dtype: float64



Both train and test dataset has same equal split in output value


```python
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```


```python
housing = strat_train_set.copy()
```


```python
housing.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12655</th>
      <td>-121.46</td>
      <td>38.52</td>
      <td>29.00</td>
      <td>3,873.00</td>
      <td>797.00</td>
      <td>2,237.00</td>
      <td>706.00</td>
      <td>2.17</td>
      <td>72,100.00</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>15502</th>
      <td>-117.23</td>
      <td>33.09</td>
      <td>7.00</td>
      <td>5,320.00</td>
      <td>855.00</td>
      <td>2,015.00</td>
      <td>768.00</td>
      <td>6.34</td>
      <td>279,600.00</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>-119.04</td>
      <td>35.37</td>
      <td>44.00</td>
      <td>1,618.00</td>
      <td>310.00</td>
      <td>667.00</td>
      <td>300.00</td>
      <td>2.88</td>
      <td>82,700.00</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>14053</th>
      <td>-117.13</td>
      <td>32.75</td>
      <td>24.00</td>
      <td>1,877.00</td>
      <td>519.00</td>
      <td>898.00</td>
      <td>483.00</td>
      <td>2.23</td>
      <td>112,500.00</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>20496</th>
      <td>-118.70</td>
      <td>34.28</td>
      <td>27.00</td>
      <td>3,536.00</td>
      <td>646.00</td>
      <td>1,837.00</td>
      <td>580.00</td>
      <td>4.50</td>
      <td>238,300.00</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>



#### 3.1 Feature Engineering

Creating a new feature from existing dataframe
- rooms_per_household is total number of room divided by households
- bedrooms_per_room is calculated by Total bedrooms divided by Total rooms
- population_per_household is ratio of population by households


```python
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
```


```python
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()
```

#### 3.2 Deal with missing data


Dataset has missing values, Impute missing values with median for all features


```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
```


```python
housing_num = housing.drop("ocean_proximity", axis=1)

```


```python
imputer.fit(housing_num)
```




    SimpleImputer(strategy='median')




```python
X = imputer.transform(housing_num)
```


```python
housing_tr = pd.DataFrame(X,columns=housing_num.columns,index = housing_num.index)
```


```python
housing_tr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12655</th>
      <td>-121.46</td>
      <td>38.52</td>
      <td>29.00</td>
      <td>3,873.00</td>
      <td>797.00</td>
      <td>2,237.00</td>
      <td>706.00</td>
      <td>2.17</td>
    </tr>
    <tr>
      <th>15502</th>
      <td>-117.23</td>
      <td>33.09</td>
      <td>7.00</td>
      <td>5,320.00</td>
      <td>855.00</td>
      <td>2,015.00</td>
      <td>768.00</td>
      <td>6.34</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>-119.04</td>
      <td>35.37</td>
      <td>44.00</td>
      <td>1,618.00</td>
      <td>310.00</td>
      <td>667.00</td>
      <td>300.00</td>
      <td>2.88</td>
    </tr>
    <tr>
      <th>14053</th>
      <td>-117.13</td>
      <td>32.75</td>
      <td>24.00</td>
      <td>1,877.00</td>
      <td>519.00</td>
      <td>898.00</td>
      <td>483.00</td>
      <td>2.23</td>
    </tr>
    <tr>
      <th>20496</th>
      <td>-118.70</td>
      <td>34.28</td>
      <td>27.00</td>
      <td>3,536.00</td>
      <td>646.00</td>
      <td>1,837.00</td>
      <td>580.00</td>
      <td>4.50</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15174</th>
      <td>-117.07</td>
      <td>33.03</td>
      <td>14.00</td>
      <td>6,665.00</td>
      <td>1,231.00</td>
      <td>2,026.00</td>
      <td>1,001.00</td>
      <td>5.09</td>
    </tr>
    <tr>
      <th>12661</th>
      <td>-121.42</td>
      <td>38.51</td>
      <td>15.00</td>
      <td>7,901.00</td>
      <td>1,422.00</td>
      <td>4,769.00</td>
      <td>1,418.00</td>
      <td>2.81</td>
    </tr>
    <tr>
      <th>19263</th>
      <td>-122.72</td>
      <td>38.44</td>
      <td>48.00</td>
      <td>707.00</td>
      <td>166.00</td>
      <td>458.00</td>
      <td>172.00</td>
      <td>3.18</td>
    </tr>
    <tr>
      <th>19140</th>
      <td>-122.70</td>
      <td>38.31</td>
      <td>14.00</td>
      <td>3,155.00</td>
      <td>580.00</td>
      <td>1,208.00</td>
      <td>501.00</td>
      <td>4.20</td>
    </tr>
    <tr>
      <th>19773</th>
      <td>-122.14</td>
      <td>39.97</td>
      <td>27.00</td>
      <td>1,079.00</td>
      <td>222.00</td>
      <td>625.00</td>
      <td>197.00</td>
      <td>3.13</td>
    </tr>
  </tbody>
</table>
<p>16512 rows × 8 columns</p>
</div>



#### 3.3 Encoding Categorical variables


```python
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12655</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>15502</th>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>14053</th>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>20496</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>1481</th>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>18125</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>5830</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>17989</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>4861</th>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
```




    array([[1.],
           [4.],
           [1.],
           [4.],
           [0.],
           [3.],
           [0.],
           [0.],
           [0.],
           [0.]])



OneHotEncoding is better than ordinal encoder to prevent auto correlation


```python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
```




    <16512x5 sparse matrix of type '<class 'numpy.float64'>'
    	with 16512 stored elements in Compressed Sparse Row format>




```python
housing_cat_1hot.toarray()
```




    array([[0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 1.],
           [0., 1., 0., 0., 0.],
           ...,
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.]])




```python
cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
```




    array([[0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 1.],
           [0., 1., 0., 0., 0.],
           ...,
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.]])



#### 3.4 Create Pipeline

I have written a function that can be used as a pipeline to clean data and predict the output in a few lines of code.




```python
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self): # no *args or **kargs
        return None
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
     
housing_extra_attribs = CombinedAttributesAdder().transform(housing.values)
```


```python
col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names] # get the column indices
```


```python
housing_extra_attribs.shape
```




    (16512, 12)




```python
housing_extra_attribsss = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household",'bedrooms_per_room'],
    index=housing.index)
housing_extra_attribsss.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
      <th>rooms_per_household</th>
      <th>population_per_household</th>
      <th>bedrooms_per_room</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12655</th>
      <td>-121.46</td>
      <td>38.52</td>
      <td>29.00</td>
      <td>3,873.00</td>
      <td>797.00</td>
      <td>2,237.00</td>
      <td>706.00</td>
      <td>2.17</td>
      <td>INLAND</td>
      <td>5.49</td>
      <td>3.17</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>15502</th>
      <td>-117.23</td>
      <td>33.09</td>
      <td>7.00</td>
      <td>5,320.00</td>
      <td>855.00</td>
      <td>2,015.00</td>
      <td>768.00</td>
      <td>6.34</td>
      <td>NEAR OCEAN</td>
      <td>6.93</td>
      <td>2.62</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>-119.04</td>
      <td>35.37</td>
      <td>44.00</td>
      <td>1,618.00</td>
      <td>310.00</td>
      <td>667.00</td>
      <td>300.00</td>
      <td>2.88</td>
      <td>INLAND</td>
      <td>5.39</td>
      <td>2.22</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>14053</th>
      <td>-117.13</td>
      <td>32.75</td>
      <td>24.00</td>
      <td>1,877.00</td>
      <td>519.00</td>
      <td>898.00</td>
      <td>483.00</td>
      <td>2.23</td>
      <td>NEAR OCEAN</td>
      <td>3.89</td>
      <td>1.86</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>20496</th>
      <td>-118.70</td>
      <td>34.28</td>
      <td>27.00</td>
      <td>3,536.00</td>
      <td>646.00</td>
      <td>1,837.00</td>
      <td>580.00</td>
      <td>4.50</td>
      <td>&lt;1H OCEAN</td>
      <td>6.10</td>
      <td>3.17</td>
      <td>0.18</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12655</th>
      <td>-121.46</td>
      <td>38.52</td>
      <td>29.00</td>
      <td>3,873.00</td>
      <td>797.00</td>
      <td>2,237.00</td>
      <td>706.00</td>
      <td>2.17</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>15502</th>
      <td>-117.23</td>
      <td>33.09</td>
      <td>7.00</td>
      <td>5,320.00</td>
      <td>855.00</td>
      <td>2,015.00</td>
      <td>768.00</td>
      <td>6.34</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>-119.04</td>
      <td>35.37</td>
      <td>44.00</td>
      <td>1,618.00</td>
      <td>310.00</td>
      <td>667.00</td>
      <td>300.00</td>
      <td>2.88</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>14053</th>
      <td>-117.13</td>
      <td>32.75</td>
      <td>24.00</td>
      <td>1,877.00</td>
      <td>519.00</td>
      <td>898.00</td>
      <td>483.00</td>
      <td>2.23</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>20496</th>
      <td>-118.70</td>
      <td>34.28</td>
      <td>27.00</td>
      <td>3,536.00</td>
      <td>646.00</td>
      <td>1,837.00</td>
      <td>580.00</td>
      <td>4.50</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>



### Step 4 Data Modelling


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```


```python
num_pipeline = Pipeline([
                ('imputer',SimpleImputer(strategy='median')),
                ('attrib_adder', CombinedAttributesAdder()),
                ('standrd_scal',StandardScaler()),
])
```


```python
housing_num_tr = num_pipeline.fit_transform(housing_num)
```


```python
housing_num_tr
```




    array([[-0.94135046,  1.34743822,  0.02756357, ...,  0.01739526,
             0.00622264, -0.12112176],
           [ 1.17178212, -1.19243966, -1.72201763, ...,  0.56925554,
            -0.04081077, -0.81086696],
           [ 0.26758118, -0.1259716 ,  1.22045984, ..., -0.01802432,
            -0.07537122, -0.33827252],
           ...,
           [-1.5707942 ,  1.31001828,  1.53856552, ..., -0.5092404 ,
            -0.03743619,  0.32286937],
           [-1.56080303,  1.2492109 , -1.1653327 , ...,  0.32814891,
            -0.05915604, -0.45702273],
           [-1.28105026,  2.02567448, -0.13148926, ...,  0.01407228,
             0.00657083, -0.12169672]])




```python
from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num',num_pipeline,num_attribs),
    ('cat',OneHotEncoder(),cat_attribs)
])
```


```python
housing_prepd = full_pipeline.fit_transform(housing)
```


```python
housing_prepd.shape
```




    (16512, 16)




```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepd, housing_labels)
```




    GridSearchCV(cv=5, estimator=RandomForestRegressor(random_state=42),
                 param_grid=[{'max_features': [2, 4, 6, 8],
                              'n_estimators': [3, 10, 30]},
                             {'bootstrap': [False], 'max_features': [2, 3, 4],
                              'n_estimators': [3, 10]}],
                 return_train_score=True, scoring='neg_mean_squared_error')




```python
grid_search.best_estimator_
```




    RandomForestRegressor(max_features=8, n_estimators=30, random_state=42)




```python
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

    63895.161577951665 {'max_features': 2, 'n_estimators': 3}
    54916.32386349543 {'max_features': 2, 'n_estimators': 10}
    52885.86715332332 {'max_features': 2, 'n_estimators': 30}
    60075.3680329983 {'max_features': 4, 'n_estimators': 3}
    52495.01284985185 {'max_features': 4, 'n_estimators': 10}
    50187.24324926565 {'max_features': 4, 'n_estimators': 30}
    58064.73529982314 {'max_features': 6, 'n_estimators': 3}
    51519.32062366315 {'max_features': 6, 'n_estimators': 10}
    49969.80441627874 {'max_features': 6, 'n_estimators': 30}
    58895.824998155826 {'max_features': 8, 'n_estimators': 3}
    52459.79624724529 {'max_features': 8, 'n_estimators': 10}
    49898.98913455217 {'max_features': 8, 'n_estimators': 30}
    62381.765106921855 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
    54476.57050944266 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
    59974.60028085155 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
    52754.5632813202 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
    57831.136061214274 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
    51278.37877140253 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}
    


```python
cvres['mean_test_score']
```




    array([-4.08259167e+09, -3.01580263e+09, -2.79691494e+09, -3.60904984e+09,
           -2.75572637e+09, -2.51875938e+09, -3.37151349e+09, -2.65424040e+09,
           -2.49698135e+09, -3.46871820e+09, -2.75203022e+09, -2.48990912e+09,
           -3.89148462e+09, -2.96769673e+09, -3.59695268e+09, -2.78304395e+09,
           -3.34444030e+09, -2.62947213e+09])




```python
cvres = grid_search.cv_results_
for meanscore,parms in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-meanscore),parms)
```

    63895.161577951665 {'max_features': 2, 'n_estimators': 3}
    54916.32386349543 {'max_features': 2, 'n_estimators': 10}
    52885.86715332332 {'max_features': 2, 'n_estimators': 30}
    60075.3680329983 {'max_features': 4, 'n_estimators': 3}
    52495.01284985185 {'max_features': 4, 'n_estimators': 10}
    50187.24324926565 {'max_features': 4, 'n_estimators': 30}
    58064.73529982314 {'max_features': 6, 'n_estimators': 3}
    51519.32062366315 {'max_features': 6, 'n_estimators': 10}
    49969.80441627874 {'max_features': 6, 'n_estimators': 30}
    58895.824998155826 {'max_features': 8, 'n_estimators': 3}
    52459.79624724529 {'max_features': 8, 'n_estimators': 10}
    49898.98913455217 {'max_features': 8, 'n_estimators': 30}
    62381.765106921855 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
    54476.57050944266 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
    59974.60028085155 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
    52754.5632813202 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
    57831.136061214274 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
    51278.37877140253 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}
    


```python
cat_enco = full_pipeline.named_transformers_['cat']
```


```python
cat_attrib = list(cat_enco.categories_[0])
```


```python
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

```


```python
attributes = num_attribs +  extra_attribs + cat_attrib
```


```python
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
```




    array([6.96542523e-02, 6.04213840e-02, 4.21882202e-02, 1.52450557e-02,
           1.55545295e-02, 1.58491147e-02, 1.49346552e-02, 3.79009225e-01,
           5.47789150e-02, 1.07031322e-01, 4.82031213e-02, 6.79266007e-03,
           1.65706303e-01, 7.83480660e-05, 1.52473276e-03, 3.02816106e-03])




```python
sorted(zip(feature_importances,attributes),reverse= True)
```




    [(0.3790092248170967, 'median_income'),
     (0.16570630316895876, 'INLAND'),
     (0.10703132208204355, 'pop_per_hhold'),
     (0.06965425227942929, 'longitude'),
     (0.0604213840080722, 'latitude'),
     (0.054778915018283726, 'rooms_per_hhold'),
     (0.048203121338269206, 'bedrooms_per_room'),
     (0.04218822024391753, 'housing_median_age'),
     (0.015849114744428634, 'population'),
     (0.015554529490469328, 'total_bedrooms'),
     (0.01524505568840977, 'total_rooms'),
     (0.014934655161887772, 'households'),
     (0.006792660074259966, '<1H OCEAN'),
     (0.0030281610628962747, 'NEAR OCEAN'),
     (0.0015247327555504937, 'NEAR BAY'),
     (7.834806602687504e-05, 'ISLAND')]




```python
fina_model = grid_search.best_estimator_
```


```python
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
```


```python
X_test_prepared = full_pipeline.transform(X_test)
```


```python
final_predictions = fina_model.predict(X_test_prepared)
```


```python
from sklearn.metrics import mean_squared_error

```


```python
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
```


```python
final_rmse
```




    47873.26095812988




```python
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
```


```python
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
```




    array([45893.36082829, 49774.46796717])




```python
m = len(squared_errors)
mean = squared_errors.mean()

zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)
```




    (45893.954011012866, 49773.92103065016)




```python

```


```python
full_pipeline_with_predictor = Pipeline([
            ('prep',full_pipeline),
            ('linear',LinearRegression())
])
```


```python
full_pipeline_with_predictor.fit(housing,housing_labels)
```




    Pipeline(steps=[('prep',
                     ColumnTransformer(transformers=[('num',
                                                      Pipeline(steps=[('imputer',
                                                                       SimpleImputer(strategy='median')),
                                                                      ('attrib_adder',
                                                                       CombinedAttributesAdder()),
                                                                      ('standrd_scal',
                                                                       StandardScaler())]),
                                                      ['longitude', 'latitude',
                                                       'housing_median_age',
                                                       'total_rooms',
                                                       'total_bedrooms',
                                                       'population', 'households',
                                                       'median_income']),
                                                     ('cat', OneHotEncoder(),
                                                      ['ocean_proximity'])])),
                    ('linear', LinearRegression())])




```python
ful_tet = full_pipeline_with_predictor.predict(X_test)
```


```python
final_mse = mean_squared_error(y_test, ful_tet)
final_rmse = np.sqrt(final_mse)
```


```python
final_rmse
```




    66913.44191320929




```python
my_model = full_pipeline_with_predictor
```


```python
import joblib
#save model
joblib.dump(my_model, "my_model.pkl") 
#load model
my_model_loaded = joblib.load("my_model.pkl") 
```


```python
my_model_loaded
```




    Pipeline(steps=[('prep',
                     ColumnTransformer(transformers=[('num',
                                                      Pipeline(steps=[('imputer',
                                                                       SimpleImputer(strategy='median')),
                                                                      ('attrib_adder',
                                                                       CombinedAttributesAdder()),
                                                                      ('standrd_scal',
                                                                       StandardScaler())]),
                                                      ['longitude', 'latitude',
                                                       'housing_median_age',
                                                       'total_rooms',
                                                       'total_bedrooms',
                                                       'population', 'households',
                                                       'median_income']),
                                                     ('cat', OneHotEncoder(),
                                                      ['ocean_proximity'])])),
                    ('linear', LinearRegression())])



### Conclusion
The root mean squared error (RMSE) for a RandomForestRegressor model was $47873.26. I did not use any other model to improve the RMSE, as my only goal was to use the Pipeline to create an output in a single run.




```python

```
