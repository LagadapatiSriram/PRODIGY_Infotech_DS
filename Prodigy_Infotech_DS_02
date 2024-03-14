#loading datasets and necessary lbraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
titanic_df = pd.read_csv('/content/titanic.csv');

# Display the first few rows of the dataset
titanic_df.head()
# gender distribution of passengers
sns.countplot(x='Sex', data=titanic_df)
plt.title('Gender Distribution')
plt.show()
# let's check for the missing values
titanic_df.isnull().sum()
# For instance, to impute missing ages, you can use the median age:
median_age = titanic_df['Age'].median()
titanic_df['Age'].fillna(median_age, inplace=True)
# Statistics of numericals
titanic_df.describe()
# distribution of passengers by class
sns.countplot(x='Pclass', data=titanic_df)
plt.title('Class Distribution of Passengers')
plt.show()
# relationship between survival and passenger class:
sns.countplot(x='Pclass', hue='Survived', data=titanic_df)
plt.title('Survival by type of Passenger Class')
plt.show()
# survival rate by gender
sns.countplot(x='Sex', hue='Survived', data=titanic_df)
plt.title("Survival by Gender")
plt.show()
# Age Distribution of passengers that survived
sns.histplot(x='Age', data=titanic_df[titanic_df['Survived'] == 1], bins=30, kde=True, color='green')
plt.title('Age Distribution of Survived Passengers')
plt.show()
# Age Distribution of passengers who not survived
sns.histplot(x='Age', data=titanic_df[titanic_df['Survived'] == 0], bins=30, kde=True, color='red')
plt.title('Age Distribution of Non-Survived Passengers')
plt.show()
# correlation heatmap:
correlation_matrix = titanic_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('correlation heatmap')
plt.show()
