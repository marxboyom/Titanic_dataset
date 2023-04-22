# Titanic_dataset

# Data Manipulation Libraries
import numpy as np
import pandas as pd

# Data Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score ,precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import  classification_report, confusion_matrix

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

#Lire les données csv avec comme première colonne PassengerID
test_df = pd.read_csv('/Users/marxboyom/Desktop/NEOMA BS/Gestion des BDD & Business Analytics/Thierry Vaillaud/titanic/test.csv')
train_df = pd.read_csv('/Users/marxboyom/Desktop/NEOMA BS/Gestion des BDD & Business Analytics/Thierry Vaillaud/titanic/train.csv')

#Nombre de lignes des deux jeux de données training et testing
rows_in_training_data = train_df.shape[0]
rows_in_testing_data = test_df.shape[0]

print(f'Rows in training data: {rows_in_training_data}.')
print(f'Rows in testing data: {rows_in_testing_data}.')
print(f'Total data: {rows_in_training_data + rows_in_testing_data}')

#Nombre de valeurs nulles par colonne
null_columns = train_df.columns[train_df.isnull().any()]
print(train_df[null_columns].isnull().sum())

#Nombre total de valeur nulles du data_train
null_columns = test_df.columns[test_df.isnull().any()]
print(test_df[null_columns].isnull().sum())

#Affichage des informations du data_train
train_df.info()

#Description du jeu de données
train_df.describe()

#corrélation entre les colonnes du dataset
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', ax=ax)
fig.suptitle('Correlations between data')
plt.show()

#Pourcentage de corrélation entre les survivants et le sex des passagers du titanic
sex_impact_on_survive = train_df[['Sex', 'Survived']].groupby('Sex')['Survived'].mean()
sex_impact_on_survive
fig, ax = plt.subplots(figsize=(8,6))
sns.barplot(x='Sex', y='Survived', data=sex_impact_on_survive.reset_index())
fig.suptitle('Correlations between sex and survived')
plt.show()

#correlation entre les places de tickets et les survivants :
fig, ax = plt.subplots(figsize=(8,6))
sns.barplot(x='Pclass', y='Survived', data=pclass_impact_on_survive.reset_index())
fig.suptitle('Correlations between ticket class and survived')
plt.show()

#liste des colonnes catégorielles et numériques :
categorical_cols = [cname for cname in train_df.columns if
                    train_df[cname].dtype == "object"]

numerical_cols = [cname for cname in train_df.columns if 
                train_df[cname].dtype in ['int64', 'float64']]

print(f'We have {len(categorical_cols)} categorical columns: {categorical_cols}')
print(f'We have {len(numerical_cols)} numerical columns: {numerical_cols}')



