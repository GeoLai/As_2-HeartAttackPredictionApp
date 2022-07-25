# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 09:36:46 2022

@author: Lai Kar Wei
"""

import os 
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def cramers_corrected_stat(con_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(con_matrix)[0]
    n = con_matrix.sum()
    phi2 = chi2/n
    r,k = con_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%% Constant paths
CSV_PATH = os.path.join(os.getcwd(), 'dataset', 'heart.csv')
BEST_ESTIMATOR_SAVE_PATH = os.path.join(os.getcwd(), 'model', 'best_estimator.pkl')

#%% Step 1 Data Loading
df = pd.read_csv(CSV_PATH)
print(df)

#%% Step 2 Data Visualization & Inspection

df.info() # info on features entries
df.describe().T # summary statistics of all the features
df.shape # shape of the data (rows, columns)
df.isna().sum() # sum of NaN in the columns
df.duplicated().sum() # any duplicate in the observation

df.drop_duplicates(inplace=True) #drop duplicate observation without making any copy of the DataFrame
df.duplicated().sum() # re-check the duplicate in the observation 

df.boxplot(rot=45) # statistical summary over boxplot indicating outliers
#separating categorical and continuous list
# Categorical features
cat_col = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall', 'output'] 

# Continuous features
cont_col = list(df.drop(labels=cat_col, axis=1)) # removing the cat_col to get cont_col, axis=1 column wise

# visualization graphs over features
#cat_col
for i in cat_col:
    plt.figure()
    sns.countplot(df[i])
    plt.show()

#aggregating target 'output' with all categorical features
df.groupby(['output', 'sex']).agg({'sex':'count'}).plot(kind='bar')
df.groupby(['output', 'cp']).agg({'cp':'count'}).plot(kind='bar')
df.groupby(['output', 'fbs']).agg({'fbs':'count'}).plot(kind='bar')
df.groupby(['output', 'restecg']).agg({'restecg':'count'}).plot(kind='bar')
df.groupby(['output', 'exng']).agg({'exng':'count'}).plot(kind='bar')
df.groupby(['output', 'slp']).agg({'slp':'count'}).plot(kind='bar')
df.groupby(['output', 'caa']).agg({'caa':'count'}).plot(kind='bar')
df.groupby(['output', 'thall']).agg({'thall':'count'}).plot(kind='bar')

#cont_col
for i in cont_col:
    plt.figure()
    sns.displot(df[i])
    plt.show()

#%% Step 3 Data cleaning
#null values changed to NaN
df['caa'] = df['caa'].replace(4, np.nan)
df['thall'] = df['thall'].replace(0, np.nan)

df.isna().sum()

#computing NaN with columns' median value
df['caa'] = df['caa'].fillna(df['caa'].median())
df['thall'] = df['thall'].fillna(df['thall'].median())

#%% Step 4 Features Selection

# to select most correlated features with target 'output' in categorical list to be selected for training, 
# correlation higher than 0.6 to be selected
for i in cat_col:
    print(i)
    con_matrix = pd.crosstab(df[i], df['output']).to_numpy()
    print(cramers_corrected_stat(con_matrix))

# low correlation in categorical features, < 0.6
# no features to be selected from categorical list

# to select most correlated features with target 'output' in continuous list to be selected for training, 
# correlation higher than 0.6 to be selected
for i in cont_col:
    print(i)
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[i], axis=1), df['output'])
    print(lr.score(np.expand_dims(df[i], axis=-1), df['output']))

#%% Step 5 Data Preprocessing
# 'age', 'thalachh', 'oldpeak' to be selected for training - > 0.6

X = df.loc[:, ['age', 'thalachh', 'oldpeak']]
y = df['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=(123))

#%% Model development

# machine learning pipelines
#Logistic Regression
pipeline_mms_lr = Pipeline([('Min_Max_Scaler', MinMaxScaler()),
                            ('Logistic_Classifier', LogisticRegression())
                            ])

pipeline_ss_lr = Pipeline([('Min_Max_Scaler', StandardScaler()),
                            ('Logistic_Classifier', LogisticRegression())
                            ])

#KNN
pipeline_mms_knn = Pipeline([('Min_Max_Scaler', MinMaxScaler()),
                            ('KNN_Classifier', KNeighborsClassifier())
                            ])

pipeline_ss_knn = Pipeline([('Min_Max_Scaler', StandardScaler()),
                            ('KNN_Classifier', KNeighborsClassifier())
                            ])

#Decision Tree
pipeline_mms_dt = Pipeline([('Min_Max_Scaler', MinMaxScaler()),
                            ('DTree_Classifier', DecisionTreeClassifier())
                            ])

pipeline_ss_dt = Pipeline([('Min_Max_Scaler', StandardScaler()),
                            ('DTree_Classifier', DecisionTreeClassifier())
                            ])

#Random Forest
pipeline_mms_rf = Pipeline([('Min_Max_Scaler', MinMaxScaler()),
                            ('RF_Classifier', RandomForestClassifier())
                            ])

pipeline_ss_rf = Pipeline([('Min_Max_Scaler', StandardScaler()),
                            ('RF_Classifier', RandomForestClassifier())
                            ])

#Gradient Boosting
pipeline_mms_gb = Pipeline([('Min_Max_Scaler', MinMaxScaler()),
                            ('GB_Classifier', GradientBoostingClassifier())
                            ])

pipeline_ss_gb = Pipeline([('Min_Max_Scaler', StandardScaler()),
                            ('GB_Classifier', GradientBoostingClassifier())
                            ])

#SVC
pipeline_mms_svc = Pipeline([('Min_Max_Scaler', MinMaxScaler()),
                            ('SVC_Classifier', SVC())
                            ])

pipeline_ss_svc = Pipeline([('Min_Max_Scaler', StandardScaler()),
                            ('SVC_Classifier', SVC())
                            ])

#create a list of pipelines
pipelines = [pipeline_mms_lr, pipeline_ss_lr, pipeline_mms_knn, pipeline_ss_knn,
             pipeline_mms_dt, pipeline_ss_dt, pipeline_mms_rf, pipeline_ss_rf,
             pipeline_mms_gb, pipeline_ss_gb, pipeline_mms_svc, pipeline_ss_svc]

# learn with pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)

best_accuracy = 0

# getting the best pipeline through scoring
for i, pipe in enumerate(pipelines):
    if pipe.score(X_test, y_test) > best_accuracy:
        best_accuracy = pipe.score(X_test, y_test)
        best_pipeline = pipe

print('The best scaler and classifier for Heart data is {} with accuracy of {}'.
      format(best_pipeline,best_accuracy))

#%% Cross Validation with GridSearchCV
# using Logistic Regression pipeline with MinMaxScaler as giving the best result with 74%

# parameters setting
params = {'Logistic_Classifier__penalty': ['l1', 'l2'],
          'Logistic_Classifier__C': np.logspace(-3, 3, 6),
          'Logistic_Classifier__solver': ['newton-cg', 'lbfgs', 'liblinear']
          }

grid_search = GridSearchCV(pipeline_mms_lr, param_grid=params, scoring='accuracy', cv=5)

grid = grid_search.fit(X_train, y_train)
grid_search.score(X_test, y_test)

print("Tuned Hyperparameters:", grid_search.best_params_) # best parameter used for CV
print("Accuracy:", grid_search.best_score_) # best accuracy

#%% Model Saving
with open(BEST_ESTIMATOR_SAVE_PATH, 'wb') as file:
    pickle.dump(grid.best_estimator_, file)

#%% Model Analysis
y_pred = grid.predict(X_test)
cr = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

labels = ['0', '1']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

print(cr)
print(cm)
