import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#load the data from csv file to PandasDataFrame
titanic_data = pd.read_csv("/content/train.csv")
#printing first five rows of the dataframe
titanic_data.head()
titanic_data.shape
titanic_data.info()
titanic_data.isnull().sum()
titanic_data = titanic_data.drop(columns='Cabin',axis=1)
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
print(titanic_data['Embarked'].mode())
print(titanic_data['Embarked'].mode()[0])
#replacing the missing values in 'Embarked' column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data.describe()
#finding the no.of people survived and not survived
titanic_data['Survived'].value_counts()
sns.set()
making a countplot for "Survived"coloumn
sns.countplot(x='Survived', data=titanic_data)
