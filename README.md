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

#préparation des données du set train_df
data_df = train_df.append(test_df)
qty_train_rows = train_df.shape[0]
data_df['Title'] = data_df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
data_df['Title'].value_counts()

mapping = {
    'Col': 'Mr', 
    'Mlle': 'Miss', 
    'Major': 'Mr', 
    'Ms': 'Miss',
    'Lady': 'Mrs', 
    'Sir': 'Mr',
    'Mme': 'Miss',
    'Don': 'Mr', 
    'Capt': 'Mr', 
    'the Countess': 'Mrs', 
    'Jonkheer': 'Mr',
    'Dona': 'Mrs'
}

data_df.replace({'Title': mapping}, inplace=True)

data_df['Title'].value_counts()

#Pourcentage de corrélation entre les survivants et le nouveau mapping
title_impact_on_survive = data_df[~data_df['Survived'].isna()][['Title', 'Survived']].groupby('Title')['Survived'].mean()
title_impact_on_survive
fig, ax = plt.subplots(figsize=(8,6))
sns.barplot(x='Title', y='Survived', data=title_impact_on_survive.reset_index())
fig.suptitle('Correlations between title and survived')
plt.show()

#histogramme de l'âge:
sns.displot(data_df['Age'], bins=20, kde=False)
plt.show()

#histogramme du prix d'achat des billets:
sns.displot(data_df['Fare'], bins=5, kde=False)
plt.show()

#Choix du modele de prédiction :
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'XGBClassifier':XGBClassifier(),
    'GradientBoostingClassifier':GradientBoostingClassifier(),
    'AdaBoostClassifier':AdaBoostClassifier()
    
}

# Train and evaluate each model using cross-validation
for name, model in models.items():
    scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
    print(f"{name} accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
    
    # Fit the model to the full training set and make predictions on the test set
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    # Evaluate the model on the test set
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1-score: {f1:.3f}")
    print()
    
    gbc= GradientBoostingClassifier()
scores = cross_val_score(gbc, x_train, y_train, cv=5, scoring='accuracy')
print(f"{gbc} accuracy: {scores}")
    
# Fit the model to the full training set and make predictions on the test set
gbc.fit(x_train, y_train)
y_pred = gbc.predict(x_test)

# Evaluate the model on the test set
acc = accuracy_score(y_test, y_pred)

#prédiction :
test_pred = gbc.predict(test)
submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': test_pred})
submission.to_csv('submission.csv', index=False)
print (acc)




