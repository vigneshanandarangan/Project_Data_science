---
jupyter:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.13
  nbformat: 4
  nbformat_minor: 5
---

::: {#297ff318-858a-4ce3-81f2-9a997cedc32f .cell .markdown}
## California Housing Prices
:::

::: {#6f7db3a1-109a-42f3-90cc-9249e3824854 .cell .markdown}
### Table of Contents

1.  [Chapter 1 - Project Overview](#Project-Overview)
2.  [Chapter 2 - Step 1: Problem Definition]()
3.  [Chapter 3 - Step 2: Data Gathering](#Step-2-Data-Gathering)
4.  [Chapter 4 - Step 3: Data Preparation](#Step-3-Data-Preperation)
5.  [Chapter 5 - Step 4: Data Modelling](#Step-4-Data-Modelling)
6.  [Chapter 6 - Conclusion](#Conclusion)
:::

::: {#52051f59-8575-47c9-ad56-357b8d26404e .cell .markdown}
### Project Overview
:::

::: {#e63dfcbc-cd72-4ea8-86d3-93698bdfe132 .cell .markdown tags="[]"}
#### Objectives

-   There are a large number of factors that can affect the value of a
    house property (eg. location, size, condition, time), these factors
    can change quite substantially from one property to another
-   The housing market itself is quite a volatile industry, and is quite
    dependent on demand and supply fluctuations, not to even mention
    economic factors such as interest rates & inflation, so its quite a
    challenge to predict the price variation over time
-   It\'s also quite challenging to predict housing prices due to the
    limited data that is available, most datasets contain a limited
    number of features related to each property, such is why feature
    engineering is quite important
-   As a result, it is generally quite difficult to accurately predict
    property prices that take into account all the factors that
    influence it
-   The California housing dataaset contains different house related
    attributes for properties located in California
:::

::: {#e8f376b8-0693-4ffa-8da1-a2b0c88572e5 .cell .markdown}
#### STUDY AIM

-   The aim is to model is to predict the median_house_value which is
    our target variable
-   Overcome missing data with a basic SimpleImputer Median function
-   Understand how to turn a simple model into your own sklearn
    comparible class, our aim won\'t be to create the most perfect model
-   This is a regression type problem i have tried to built pipeline and
    predict the output no visualization added
:::

::: {#2fd3c8dd-5fee-40fb-b4e4-d761a67c2f10 .cell .code execution_count="2"}
``` python
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

::: {.output .stream .stdout}
    Python version: 3.9.13 (main, Aug 25 2022, 23:51:50) [MSC v.1916 64 bit (AMD64)]
    pandas version: 1.4.4
    matplotlib version: 3.5.2
    NumPy version: 1.21.5
    SciPy version: 1.9.1
    IPython version: 7.31.1
    scikit-learn version: 1.0.2
    -------------------------
:::
:::

::: {#a8f976f2-bbbd-4ee7-b552-416e510b60c5 .cell .code execution_count="10"}
``` python
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
:::

::: {#1ff0b991-dabd-4f69-980a-a2136c1dc40f .cell .markdown}
### Step 1 Problem Definition
:::

::: {#5ccf4425-643d-4939-8330-6ddc8e984ee0 .cell .markdown}
We collected information on the variables using all the block groups in
California from the 1990 Cens us. In this sample a block group on
average includes 1425.5 individuals living in a geographically compact
area. Naturally, the geographical area included varies inversely with
the population density. We computed distances among the centroids of
each block group as measured in latitude and longitude. We excluded all
the block groups reporting zero entries for the independent and
dependent variables. The final data contained 20,640 observations on 9
variables. The dependent variable is ln(median house value).
:::

::: {#28fda007-308d-4535-83ec-e47ba268149b .cell .markdown}
### Step 2 Data Gathering

We use pandas.read_csv() function to read the csv file. In the bracket,
we put the file path along with a quotation mark, so that pandas will
read the file into a data frame from that address. The file path can be
either an URL or your local file address.

Because the data has first row has header, we can add an argument
headers = 1 inside the read_csv() method, so that pandas will
automatically set the first row as a header. You can also assign the
dataset to any variable you create.

The dataset can be downloaded from
[here](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
:::

::: {#ee3845d9-4d33-42eb-a411-16155ffe7d11 .cell .code execution_count="11" tags="[]"}
``` python
housing = pd.read_csv('D:\\python_dir_Jupyter\\Project_resume\\Data_science_project\\4_Housing_pipeline\\housing.csv')
housing.shape
```

::: {.output .execute_result execution_count="11"}
    (20640, 10)
:::
:::

::: {#0ea9a8e7-4f8a-4779-98d2-efdf3317e68a .cell .code execution_count="12"}
``` python
housing.head()
```

::: {.output .execute_result execution_count="12"}
```{=html}
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
```
:::
:::

::: {#a519ebdc-0f7a-45ac-8e2c-937d70f996ba .cell .markdown}
median_house_value is the output we should predict
:::

::: {#c07b7c8e-19ab-46e0-8a9a-e83dfb65ad14 .cell .markdown}
### Step 3 Data Preperation

The data was pre-processed, so I only focused on cleaning it up further.
:::

::: {#f082e284-daf1-4e7a-afe1-9b4f00e9bbdb .cell .code execution_count="13"}
``` python
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
```
:::

::: {#83e2166c-21f3-483c-9740-d5949bb8d989 .cell .code execution_count="15"}
``` python
housing["income_cat"].value_counts()
```

::: {.output .execute_result execution_count="15"}
    3    7236
    2    6581
    4    3639
    5    2362
    1     822
    Name: income_cat, dtype: int64
:::
:::

::: {#bd0b6ef6-2fe6-46f0-ac42-c450805959b6 .cell .markdown}
Splitting the dataset in equal ration of income_cat in train and test
dataset
:::

::: {#b048ef09-d971-40b2-8f4b-8e78ccff6634 .cell .code execution_count="13"}
``` python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```
:::

::: {#b1c4afa0-b0fc-4f97-88ca-a8074c605dbf .cell .code execution_count="11"}
``` python
strat_train_set
```

::: {.output .execute_result execution_count="11"}
```{=html}
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
      <td>29.0</td>
      <td>3873.0</td>
      <td>797.0</td>
      <td>2237.0</td>
      <td>706.0</td>
      <td>2.1736</td>
      <td>72100.0</td>
      <td>INLAND</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15502</th>
      <td>-117.23</td>
      <td>33.09</td>
      <td>7.0</td>
      <td>5320.0</td>
      <td>855.0</td>
      <td>2015.0</td>
      <td>768.0</td>
      <td>6.3373</td>
      <td>279600.0</td>
      <td>NEAR OCEAN</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>-119.04</td>
      <td>35.37</td>
      <td>44.0</td>
      <td>1618.0</td>
      <td>310.0</td>
      <td>667.0</td>
      <td>300.0</td>
      <td>2.8750</td>
      <td>82700.0</td>
      <td>INLAND</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14053</th>
      <td>-117.13</td>
      <td>32.75</td>
      <td>24.0</td>
      <td>1877.0</td>
      <td>519.0</td>
      <td>898.0</td>
      <td>483.0</td>
      <td>2.2264</td>
      <td>112500.0</td>
      <td>NEAR OCEAN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20496</th>
      <td>-118.70</td>
      <td>34.28</td>
      <td>27.0</td>
      <td>3536.0</td>
      <td>646.0</td>
      <td>1837.0</td>
      <td>580.0</td>
      <td>4.4964</td>
      <td>238300.0</td>
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
      <td>14.0</td>
      <td>6665.0</td>
      <td>1231.0</td>
      <td>2026.0</td>
      <td>1001.0</td>
      <td>5.0900</td>
      <td>268500.0</td>
      <td>&lt;1H OCEAN</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12661</th>
      <td>-121.42</td>
      <td>38.51</td>
      <td>15.0</td>
      <td>7901.0</td>
      <td>1422.0</td>
      <td>4769.0</td>
      <td>1418.0</td>
      <td>2.8139</td>
      <td>90400.0</td>
      <td>INLAND</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19263</th>
      <td>-122.72</td>
      <td>38.44</td>
      <td>48.0</td>
      <td>707.0</td>
      <td>166.0</td>
      <td>458.0</td>
      <td>172.0</td>
      <td>3.1797</td>
      <td>140400.0</td>
      <td>&lt;1H OCEAN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19140</th>
      <td>-122.70</td>
      <td>38.31</td>
      <td>14.0</td>
      <td>3155.0</td>
      <td>580.0</td>
      <td>1208.0</td>
      <td>501.0</td>
      <td>4.1964</td>
      <td>258100.0</td>
      <td>&lt;1H OCEAN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19773</th>
      <td>-122.14</td>
      <td>39.97</td>
      <td>27.0</td>
      <td>1079.0</td>
      <td>222.0</td>
      <td>625.0</td>
      <td>197.0</td>
      <td>3.1319</td>
      <td>62700.0</td>
      <td>INLAND</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>16512 rows × 11 columns</p>
</div>
```
:::
:::

::: {#5d32f7d7-293c-4485-9a65-862775718e2e .cell .code execution_count="12"}
``` python
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
```

::: {.output .execute_result execution_count="12"}
    3    0.350533
    2    0.318798
    4    0.176357
    5    0.114341
    1    0.039971
    Name: income_cat, dtype: float64
:::
:::

::: {#b3c34ac5-d125-4f61-b519-231e34225c9a .cell .code execution_count="13"}
``` python
strat_train_set["income_cat"].value_counts() / len(strat_train_set)
```

::: {.output .execute_result execution_count="13"}
    3    0.350594
    2    0.318859
    4    0.176296
    5    0.114462
    1    0.039789
    Name: income_cat, dtype: float64
:::
:::

::: {#7c542e8f-e4f4-4d37-8ea3-81b6ff997ee7 .cell .markdown}
Both train and test dataset has same equal split in output value
:::

::: {#7a003ffa-91e0-4dc8-aa11-734ce16bb7b5 .cell .code execution_count="14"}
``` python
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```
:::

::: {#c8d3faae-2a89-4ca8-9a98-39cd0a148bf7 .cell .code execution_count="15"}
``` python
housing = strat_train_set.copy()
```
:::

::: {#a827a0e4-3339-4347-b95f-63e9e8335265 .cell .code execution_count="16"}
``` python
housing.head()
```

::: {.output .execute_result execution_count="16"}
```{=html}
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
      <td>29.0</td>
      <td>3873.0</td>
      <td>797.0</td>
      <td>2237.0</td>
      <td>706.0</td>
      <td>2.1736</td>
      <td>72100.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>15502</th>
      <td>-117.23</td>
      <td>33.09</td>
      <td>7.0</td>
      <td>5320.0</td>
      <td>855.0</td>
      <td>2015.0</td>
      <td>768.0</td>
      <td>6.3373</td>
      <td>279600.0</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>-119.04</td>
      <td>35.37</td>
      <td>44.0</td>
      <td>1618.0</td>
      <td>310.0</td>
      <td>667.0</td>
      <td>300.0</td>
      <td>2.8750</td>
      <td>82700.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>14053</th>
      <td>-117.13</td>
      <td>32.75</td>
      <td>24.0</td>
      <td>1877.0</td>
      <td>519.0</td>
      <td>898.0</td>
      <td>483.0</td>
      <td>2.2264</td>
      <td>112500.0</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>20496</th>
      <td>-118.70</td>
      <td>34.28</td>
      <td>27.0</td>
      <td>3536.0</td>
      <td>646.0</td>
      <td>1837.0</td>
      <td>580.0</td>
      <td>4.4964</td>
      <td>238300.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#91dd477b-f2b0-4315-842d-9de8f179339f .cell .markdown}
#### 3.1 Feature Engineering {#31-feature-engineering}
:::

::: {#6a30f725-59cb-4698-8e68-d17eb9b96399 .cell .markdown}
Creating a new feature from existing dataframe

-   rooms_per_household is total number of room divided by households
-   bedrooms_per_room is calculated by Total bedrooms divided by Total
    rooms
-   population_per_household is ratio of population by households
:::

::: {#1ce18327-c1a6-4ac8-9ba8-1266a0ebc2d2 .cell .code}
``` python
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
```
:::

::: {#11a51b72-4069-41c8-97e6-69ce59289087 .cell .code execution_count="18"}
``` python
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()
```
:::

::: {#d1e98b63-c003-4475-98d6-d100a689f4a9 .cell .markdown}
#### 3.2 Deal with missing data {#32-deal-with-missing-data}
:::

::: {#bf835110-0aa7-47bf-b1ec-0d535546f02c .cell .markdown}
Dataset has missing values, Impute missing values with median for all
features
:::

::: {#117a8f3f-c08c-49d9-bcf6-6182d68d374d .cell .code execution_count="19"}
``` python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
```
:::

::: {#7199b757-c9db-49ae-a13f-459627afd777 .cell .code execution_count="20"}
``` python
housing_num = housing.drop("ocean_proximity", axis=1)
```
:::

::: {#99da19f1-a48b-46f9-9cd4-4efc09f281dd .cell .code execution_count="21"}
``` python
imputer.fit(housing_num)
```

::: {.output .execute_result execution_count="21"}
    SimpleImputer(strategy='median')
:::
:::

::: {#e2b20d11-00dd-470a-9bc9-7d6797bed1a8 .cell .code execution_count="22"}
``` python
X = imputer.transform(housing_num)
```
:::

::: {#fd88f4d3-063b-4229-8223-0a23eab2da88 .cell .code execution_count="23"}
``` python
housing_tr = pd.DataFrame(X,columns=housing_num.columns,index = housing_num.index)
```
:::

::: {#4caacf7c-fd9f-4d8a-aa51-48259c56cf6f .cell .code execution_count="24"}
``` python
housing_tr
```

::: {.output .execute_result execution_count="24"}
```{=html}
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
      <td>29.0</td>
      <td>3873.0</td>
      <td>797.0</td>
      <td>2237.0</td>
      <td>706.0</td>
      <td>2.1736</td>
    </tr>
    <tr>
      <th>15502</th>
      <td>-117.23</td>
      <td>33.09</td>
      <td>7.0</td>
      <td>5320.0</td>
      <td>855.0</td>
      <td>2015.0</td>
      <td>768.0</td>
      <td>6.3373</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>-119.04</td>
      <td>35.37</td>
      <td>44.0</td>
      <td>1618.0</td>
      <td>310.0</td>
      <td>667.0</td>
      <td>300.0</td>
      <td>2.8750</td>
    </tr>
    <tr>
      <th>14053</th>
      <td>-117.13</td>
      <td>32.75</td>
      <td>24.0</td>
      <td>1877.0</td>
      <td>519.0</td>
      <td>898.0</td>
      <td>483.0</td>
      <td>2.2264</td>
    </tr>
    <tr>
      <th>20496</th>
      <td>-118.70</td>
      <td>34.28</td>
      <td>27.0</td>
      <td>3536.0</td>
      <td>646.0</td>
      <td>1837.0</td>
      <td>580.0</td>
      <td>4.4964</td>
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
      <td>14.0</td>
      <td>6665.0</td>
      <td>1231.0</td>
      <td>2026.0</td>
      <td>1001.0</td>
      <td>5.0900</td>
    </tr>
    <tr>
      <th>12661</th>
      <td>-121.42</td>
      <td>38.51</td>
      <td>15.0</td>
      <td>7901.0</td>
      <td>1422.0</td>
      <td>4769.0</td>
      <td>1418.0</td>
      <td>2.8139</td>
    </tr>
    <tr>
      <th>19263</th>
      <td>-122.72</td>
      <td>38.44</td>
      <td>48.0</td>
      <td>707.0</td>
      <td>166.0</td>
      <td>458.0</td>
      <td>172.0</td>
      <td>3.1797</td>
    </tr>
    <tr>
      <th>19140</th>
      <td>-122.70</td>
      <td>38.31</td>
      <td>14.0</td>
      <td>3155.0</td>
      <td>580.0</td>
      <td>1208.0</td>
      <td>501.0</td>
      <td>4.1964</td>
    </tr>
    <tr>
      <th>19773</th>
      <td>-122.14</td>
      <td>39.97</td>
      <td>27.0</td>
      <td>1079.0</td>
      <td>222.0</td>
      <td>625.0</td>
      <td>197.0</td>
      <td>3.1319</td>
    </tr>
  </tbody>
</table>
<p>16512 rows × 8 columns</p>
</div>
```
:::
:::

::: {#4373f767-2dad-402e-a4d0-e8ee0fd8fc47 .cell .markdown}
#### 3.3 Encoding Categorical variables {#33-encoding-categorical-variables}
:::

::: {#aa815588-904a-40a6-ac9c-a3738c419b77 .cell .code execution_count="25"}
``` python
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)
```

::: {.output .execute_result execution_count="25"}
```{=html}
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
```
:::
:::

::: {#c9529c86-9b66-4d72-a3b7-71e760b7eaef .cell .code execution_count="26"}
``` python
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
```

::: {.output .execute_result execution_count="26"}
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
:::
:::

::: {#2144de0f-e177-4c6c-b775-6cd28838ea83 .cell .markdown}
OneHotEncoding is better than ordinal encoder to prevent auto
correlation
:::

::: {#ec03ebf5-3996-4e6c-82bb-0fb60e4953c0 .cell .code execution_count="27"}
``` python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
```

::: {.output .execute_result execution_count="27"}
    <16512x5 sparse matrix of type '<class 'numpy.float64'>'
    	with 16512 stored elements in Compressed Sparse Row format>
:::
:::

::: {#f27ae686-b1a9-4a0f-a2c2-d160e514fec1 .cell .code execution_count="28"}
``` python
housing_cat_1hot.toarray()
```

::: {.output .execute_result execution_count="28"}
    array([[0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 1.],
           [0., 1., 0., 0., 0.],
           ...,
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.]])
:::
:::

::: {#7076aa5d-6699-449b-bba1-4d853b0473be .cell .code execution_count="29"}
``` python
cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
```

::: {.output .execute_result execution_count="29"}
    array([[0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 1.],
           [0., 1., 0., 0., 0.],
           ...,
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.]])
:::
:::

::: {#36dc587a-5e1e-42ab-945d-e6daf6dd0a6b .cell .markdown}
#### 3.4 Create Pipeline {#34-create-pipeline}
:::

::: {#f0670880-6fa3-4f76-bf57-f148e2e3a8b0 .cell .markdown}
I have written a function that can be used as a pipeline to clean data
and predict the output in a few lines of code.
:::

::: {#771243a2-2e96-4c93-ad5e-2ea675122794 .cell .code execution_count="32"}
``` python
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
:::

::: {#126fe9e2-449a-473d-a5d1-ca330a93cc10 .cell .code}
``` python
col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names] # get the column indices
```
:::

::: {#5aff9963-d1ec-4b5a-b3a7-ab448afc43b5 .cell .code execution_count="33"}
``` python
housing_extra_attribs.shape
```

::: {.output .execute_result execution_count="33"}
    (16512, 12)
:::
:::

::: {#10d839d8-7c1c-48d0-bfd7-e64d62633de6 .cell .code execution_count="34"}
``` python
housing_extra_attribsss = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household",'bedrooms_per_room'],
    index=housing.index)
housing_extra_attribsss.head()
```

::: {.output .execute_result execution_count="34"}
```{=html}
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
      <td>29.0</td>
      <td>3873.0</td>
      <td>797.0</td>
      <td>2237.0</td>
      <td>706.0</td>
      <td>2.1736</td>
      <td>INLAND</td>
      <td>5.485836</td>
      <td>3.168555</td>
      <td>0.205784</td>
    </tr>
    <tr>
      <th>15502</th>
      <td>-117.23</td>
      <td>33.09</td>
      <td>7.0</td>
      <td>5320.0</td>
      <td>855.0</td>
      <td>2015.0</td>
      <td>768.0</td>
      <td>6.3373</td>
      <td>NEAR OCEAN</td>
      <td>6.927083</td>
      <td>2.623698</td>
      <td>0.160714</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>-119.04</td>
      <td>35.37</td>
      <td>44.0</td>
      <td>1618.0</td>
      <td>310.0</td>
      <td>667.0</td>
      <td>300.0</td>
      <td>2.875</td>
      <td>INLAND</td>
      <td>5.393333</td>
      <td>2.223333</td>
      <td>0.191595</td>
    </tr>
    <tr>
      <th>14053</th>
      <td>-117.13</td>
      <td>32.75</td>
      <td>24.0</td>
      <td>1877.0</td>
      <td>519.0</td>
      <td>898.0</td>
      <td>483.0</td>
      <td>2.2264</td>
      <td>NEAR OCEAN</td>
      <td>3.886128</td>
      <td>1.859213</td>
      <td>0.276505</td>
    </tr>
    <tr>
      <th>20496</th>
      <td>-118.7</td>
      <td>34.28</td>
      <td>27.0</td>
      <td>3536.0</td>
      <td>646.0</td>
      <td>1837.0</td>
      <td>580.0</td>
      <td>4.4964</td>
      <td>&lt;1H OCEAN</td>
      <td>6.096552</td>
      <td>3.167241</td>
      <td>0.182692</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#1d80264c-144a-428a-b1bf-25a0db315909 .cell .code execution_count="35"}
``` python
housing.head()
```

::: {.output .execute_result execution_count="35"}
```{=html}
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
      <td>29.0</td>
      <td>3873.0</td>
      <td>797.0</td>
      <td>2237.0</td>
      <td>706.0</td>
      <td>2.1736</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>15502</th>
      <td>-117.23</td>
      <td>33.09</td>
      <td>7.0</td>
      <td>5320.0</td>
      <td>855.0</td>
      <td>2015.0</td>
      <td>768.0</td>
      <td>6.3373</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>-119.04</td>
      <td>35.37</td>
      <td>44.0</td>
      <td>1618.0</td>
      <td>310.0</td>
      <td>667.0</td>
      <td>300.0</td>
      <td>2.8750</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>14053</th>
      <td>-117.13</td>
      <td>32.75</td>
      <td>24.0</td>
      <td>1877.0</td>
      <td>519.0</td>
      <td>898.0</td>
      <td>483.0</td>
      <td>2.2264</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>20496</th>
      <td>-118.70</td>
      <td>34.28</td>
      <td>27.0</td>
      <td>3536.0</td>
      <td>646.0</td>
      <td>1837.0</td>
      <td>580.0</td>
      <td>4.4964</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#e0773401-389e-4ca6-b894-de4be4024c08 .cell .markdown}
### Step 4 Data Modelling
:::

::: {#e61168f4-82a5-47b9-9be7-775bcd3b51e3 .cell .code execution_count="37"}
``` python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```
:::

::: {#05e0bfb6-2cc4-4ae7-9532-558da2fa4eba .cell .code execution_count="38"}
``` python
num_pipeline = Pipeline([
                ('imputer',SimpleImputer(strategy='median')),
                ('attrib_adder', CombinedAttributesAdder()),
                ('standrd_scal',StandardScaler()),
])
```
:::

::: {#efb8e561-489d-4c02-81a2-37fd95239ae4 .cell .code execution_count="40"}
``` python
housing_num_tr = num_pipeline.fit_transform(housing_num)
```
:::

::: {#76e5c502-7bac-4f2b-8e08-78c1596abbd8 .cell .code execution_count="41"}
``` python
housing_num_tr
```

::: {.output .execute_result execution_count="41"}
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
:::
:::

::: {#83bba76c-f1d9-47df-b8c1-3804554bf702 .cell .code execution_count="73"}
``` python
from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num',num_pipeline,num_attribs),
    ('cat',OneHotEncoder(),cat_attribs)
])
```
:::

::: {#cdb3a886-b52f-404e-b292-e847bd74de44 .cell .code execution_count="74"}
``` python
housing_prepd = full_pipeline.fit_transform(housing)
```
:::

::: {#9eaf5b80-3252-4d38-8e5d-f1e306df0013 .cell .code execution_count="75"}
``` python
housing_prepd.shape
```

::: {.output .execute_result execution_count="75"}
    (16512, 16)
:::
:::

::: {#32703f34-3bd8-49f1-b11c-c1224dc7f79a .cell .code execution_count="48"}
``` python
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

::: {.output .execute_result execution_count="48"}
    GridSearchCV(cv=5, estimator=RandomForestRegressor(random_state=42),
                 param_grid=[{'max_features': [2, 4, 6, 8],
                              'n_estimators': [3, 10, 30]},
                             {'bootstrap': [False], 'max_features': [2, 3, 4],
                              'n_estimators': [3, 10]}],
                 return_train_score=True, scoring='neg_mean_squared_error')
:::
:::

::: {#823e9e98-df15-4986-83e0-e185c5d44602 .cell .code execution_count="49"}
``` python
grid_search.best_estimator_
```

::: {.output .execute_result execution_count="49"}
    RandomForestRegressor(max_features=8, n_estimators=30, random_state=42)
:::
:::

::: {#88b2e23f-29fc-4ddf-95f5-4c0df41e5846 .cell .code execution_count="51"}
``` python
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

::: {.output .stream .stdout}
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
:::
:::

::: {#088b6e2a-8db2-489d-b20d-d5c15cabcb1c .cell .code execution_count="52"}
``` python
cvres['mean_test_score']
```

::: {.output .execute_result execution_count="52"}
    array([-4.08259167e+09, -3.01580263e+09, -2.79691494e+09, -3.60904984e+09,
           -2.75572637e+09, -2.51875938e+09, -3.37151349e+09, -2.65424040e+09,
           -2.49698135e+09, -3.46871820e+09, -2.75203022e+09, -2.48990912e+09,
           -3.89148462e+09, -2.96769673e+09, -3.59695268e+09, -2.78304395e+09,
           -3.34444030e+09, -2.62947213e+09])
:::
:::

::: {#4c4801b1-b360-4aac-8530-63ff7927d0f0 .cell .code execution_count="53"}
``` python
cvres = grid_search.cv_results_
for meanscore,parms in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-meanscore),parms)
```

::: {.output .stream .stdout}
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
:::
:::

::: {#7558cde5-5ba3-4c8c-ba9f-77aac59b687c .cell .code execution_count="54"}
``` python
cat_enco = full_pipeline.named_transformers_['cat']
```
:::

::: {#cf53fc45-6a62-44a0-9c17-02a0fd44d591 .cell .code execution_count="56"}
``` python
cat_attrib = list(cat_enco.categories_[0])
```
:::

::: {#c18abdae-c257-4253-be2c-3223688ecb20 .cell .code execution_count="59"}
``` python
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
```
:::

::: {#ce5aaba9-d93a-4973-a072-51c80dbded32 .cell .code execution_count="60"}
``` python
attributes = num_attribs +  extra_attribs + cat_attrib
```
:::

::: {#dbc11e20-e3d9-4ce5-928f-78ff70fb0cd6 .cell .code execution_count="62"}
``` python
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
```

::: {.output .execute_result execution_count="62"}
    array([6.96542523e-02, 6.04213840e-02, 4.21882202e-02, 1.52450557e-02,
           1.55545295e-02, 1.58491147e-02, 1.49346552e-02, 3.79009225e-01,
           5.47789150e-02, 1.07031322e-01, 4.82031213e-02, 6.79266007e-03,
           1.65706303e-01, 7.83480660e-05, 1.52473276e-03, 3.02816106e-03])
:::
:::

::: {#e2554a85-8075-4eae-ba7d-4a7cb1ae782d .cell .code execution_count="63"}
``` python
sorted(zip(feature_importances,attributes),reverse= True)
```

::: {.output .execute_result execution_count="63"}
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
:::
:::

::: {#2a731a05-15d1-484e-9730-66dee625d6d5 .cell .code execution_count="64"}
``` python
fina_model = grid_search.best_estimator_
```
:::

::: {#b9ef5d3d-f96c-4845-bdf2-939e5f37c7ec .cell .code execution_count="65"}
``` python
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
```
:::

::: {#ec4c720f-95c6-48c5-aa9f-96324db91be7 .cell .code execution_count="76"}
``` python
X_test_prepared = full_pipeline.transform(X_test)
```
:::

::: {#ecede9cd-9d98-45b7-9d4e-1bb1dcca864a .cell .code execution_count="78"}
``` python
final_predictions = fina_model.predict(X_test_prepared)
```
:::

::: {#f2ee5cee-0df7-4b1f-a0c9-5916e45b41ed .cell .code execution_count="79"}
``` python
from sklearn.metrics import mean_squared_error
```
:::

::: {#4463bc3b-9aed-4e29-b229-532fea5a50d4 .cell .code execution_count="80"}
``` python
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
```
:::

::: {#6ae694e7-d695-4767-860d-09505426421c .cell .code execution_count="81"}
``` python
final_rmse
```

::: {.output .execute_result execution_count="81"}
    47873.26095812988
:::
:::

::: {#ee852a22-7943-440c-962d-c50ad52a7e17 .cell .code execution_count="82"}
``` python
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
```
:::

::: {#26cf21e9-bdb7-4783-a5eb-f49530ed4815 .cell .code execution_count="83"}
``` python
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
```

::: {.output .execute_result execution_count="83"}
    array([45893.36082829, 49774.46796717])
:::
:::

::: {#c024f200-a02e-4194-8d1d-64907799630a .cell .code execution_count="86"}
``` python
m = len(squared_errors)
mean = squared_errors.mean()

zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)
```

::: {.output .execute_result execution_count="86"}
    (45893.954011012866, 49773.92103065016)
:::
:::

::: {#cb78767e-6223-47d2-9a47-cc2a8b331ebe .cell .code execution_count="88"}
``` python
```
:::

::: {#ae8f85bd-da05-4d44-b4d3-7bc06642a523 .cell .code execution_count="89"}
``` python
full_pipeline_with_predictor = Pipeline([
            ('prep',full_pipeline),
            ('linear',LinearRegression())
])
```
:::

::: {#5183dfb9-7275-41ac-bce2-f7c0f2984bf8 .cell .code execution_count="91"}
``` python
full_pipeline_with_predictor.fit(housing,housing_labels)
```

::: {.output .execute_result execution_count="91"}
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
:::
:::

::: {#d2480df6-8cbf-4fe1-8442-f4117a94ce3b .cell .code execution_count="93"}
``` python
ful_tet = full_pipeline_with_predictor.predict(X_test)
```
:::

::: {#3144a7b2-4484-46f3-9233-33ce65e28bd4 .cell .code execution_count="94"}
``` python
final_mse = mean_squared_error(y_test, ful_tet)
final_rmse = np.sqrt(final_mse)
```
:::

::: {#d914f0d1-bd31-4648-a7bc-a83bb87d2ae2 .cell .code execution_count="95"}
``` python
final_rmse
```

::: {.output .execute_result execution_count="95"}
    66913.44191320929
:::
:::

::: {#3aefd51f-54d3-40bd-9b25-b4c5f44fb31a .cell .code execution_count="96"}
``` python
my_model = full_pipeline_with_predictor
```
:::

::: {#7f74524c-9e31-4852-a3a3-adad806c7dd7 .cell .code execution_count="97"}
``` python
import joblib
#save model
joblib.dump(my_model, "my_model.pkl") 
#load model
my_model_loaded = joblib.load("my_model.pkl") 
```
:::

::: {#699c99b8-832a-4791-bd7e-ab4470284a32 .cell .code execution_count="98"}
``` python
my_model_loaded
```

::: {.output .execute_result execution_count="98"}
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
:::
:::

::: {#9a86b9fd-fb65-4e1d-beae-3aaa3ec4cdbf .cell .markdown}
### Conclusion

The root mean squared error (RMSE) for a RandomForestRegressor model was
\$47873.26. I did not use any other model to improve the RMSE, as my
only goal was to use the Pipeline to create an output in a single run.
:::

::: {#79a5ad29-85dd-4e10-a508-510b79f948c6 .cell .code}
``` python
```
:::
