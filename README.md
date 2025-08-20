<H3>ENTER YOUR NAME: DHANUSYA K</H3>
<H3>ENTER YOUR REGISTER NO: 212223230043</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 20-08-2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
#### Import libraries
~~~
from google.colab import files
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
~~~
#### Read the dataset
~~~
df=pd.read_csv("Churn_Modelling.csv")
df.head()
df.tail()
df.columns
~~~
#### Check the missing data
~~~
df.isnull().sum()
df.duplicated()
~~~
#### Assigning y
~~~
y = df.iloc[:, -1].values
print(y)
~~~
#### Check for duplicates
~~~
df.duplicated()
~~~
#### Check for outliers
~~~
df.describe()
~~~
#### Droping string values from the dataset
~~~
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
~~~
#### Checking datasets after dropping string values data from dataset
~~~
data.head()
~~~
#### Normalize the dataset
~~~
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
~~~
#### Split the dataset
~~~
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
~~~
#### Training and testing model
~~~
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
~~~
## OUTPUT:
#### Data Checking
<img width="903" height="107" alt="image" src="https://github.com/user-attachments/assets/22a60fd6-7615-4b4b-8394-ed18c2a1aa3d" />

#### Duplicates Identification
<img width="240" height="511" alt="image" src="https://github.com/user-attachments/assets/58812900-4698-4ef7-8951-5202d84a3bb6" />

#### Values of 'Y'
<img width="216" height="38" alt="image" src="https://github.com/user-attachments/assets/7ae48a2b-61b8-4510-8ef6-d846bd66cb59" />

#### Outliers
<img width="1419" height="361" alt="image" src="https://github.com/user-attachments/assets/107e1500-8369-4cdb-8fc7-bf24ac6c2fcb" />

#### Checking datasets after dropping string values data from dataset
<img width="1193" height="239" alt="image" src="https://github.com/user-attachments/assets/f19449c6-183a-47f4-b8c3-f80e700240e8" />

#### Normalize the dataset
<img width="807" height="541" alt="image" src="https://github.com/user-attachments/assets/b3c0df02-de6f-46a0-9018-1cfbd765c7ea" />

#### Split the dataset
<img width="454" height="172" alt="image" src="https://github.com/user-attachments/assets/b92c1281-eb9f-460d-af49-e17baddc0add" />

#### Training the Model
<img width="471" height="221" alt="image" src="https://github.com/user-attachments/assets/fe5c16f1-7003-4fee-9131-9453d551c246" />

#### Testing the Model
<img width="521" height="246" alt="image" src="https://github.com/user-attachments/assets/11cced0b-8a39-4828-8577-019f10c64858" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


