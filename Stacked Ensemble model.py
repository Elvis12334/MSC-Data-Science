#!/usr/bin/env python
# coding: utf-8

# # Import the Neccesary Libraries #

# In[81]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifier, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# import ace_tools as tools
from pprint import pprint
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


# In[3]:


# !pip install imbalanced-learn
from imblearn.over_sampling import SMOTE



# In[4]:


# pip install --upgrade xgboost


# In[5]:


# Load the dataset
data = pd.read_csv('/Users/apple/Desktop/DATA SCIENCE PROJECT PREPARARTION /DATA/StudentPerformanceFactors (1).csv')
(data.head(10))


# # EXPLORATORY DATA ANALYSIS AND DATA PREPROCESSING #

# In[7]:


# Check the dimensions (rows, columns) of the dataset
data.shape


# In[8]:


# Overview of the Dataset
print(data.info())


# In[9]:


print(data.dtypes)


# In[11]:


# Separate categorical and numerical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Printing the number of columns and each column in segments
print('=== NUMERICAL COLUMNS ===')
print(f"Number of numerical columns: {len(numerical_cols)}")
print("Numerical Columns:")
for col in numerical_cols:
    print(f"- {col}")

print('\n')

print('=== CATEGORICAL COLUMNS ===')
print(f"Number of categorical columns: {len(categorical_cols)}")
print("Categorical Columns:")
for col in categorical_cols:
    print(f"- {col}")


# In[11]:


# Checking for missing Values
print(data.isnull().sum())


# In[12]:


# Descriptive Statistical Summary for Numerical column
data.describe()


# In[13]:


# Descriptive statistics for categorical columns
data.describe(include='object')


# In[15]:


# Imputation for numerical data (using mean)
num_imputer = SimpleImputer(strategy='mean')
data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])

# Imputation for categorical data (using most frequent)
cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])


# In[17]:


# Checking for missing Values after preprocessing 
print(data.isnull().sum())


# In[19]:


# Cap the score at 100
data['Exam_Score'] = data['Exam_Score'].apply(lambda x: min(x, 100))


# In[21]:


# The number of duplicate rows in the DataFrame.
data.duplicated().sum()


# In[23]:


# Plotting the distribution of Exam Score
plt.figure(figsize=(10, 6))
sns.histplot(data['Exam_Score'], kde=True, bins=20)
plt.title('Distribution of Exam Scores')
plt.xlabel('Exam_Score')
plt.ylabel('Frequency')
plt.show()


# In[25]:


# Rename the 'Exam_Score' column to 'Grade'
data.rename(columns={'Exam_Score': 'Grade'}, inplace=True)

# Display the updated DataFrame
data


# In[27]:


# Replacing the values in 'Grade' with Grade Mapping
grade_mapping = {
    'credit': 0,
    'merit': 1,
    'distinction': 2
}
data['Grade'] = data['Grade'].replace(grade_mapping)
# print(data)


# In[31]:


#Transforming categorical variables to numerical variables
# Loop through all columns and encode them
for column in data.columns:
    # If the column type is object or contains fewer unique values (categorical-like)
    if data[column].dtype == 'object' or data[column].nunique() < 10:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        
(data.head(10))


# In[35]:


# Classify the Exam Scores in Specific Range
def classify_score(x):
    if x < 50:
        return 'fail'
    elif 50 <= x <= 59:
        return 'credit'
    elif 60 <= x <= 69:
        return 'merit'
    else:
        return 'distinction'

data['Grade'] = data['Grade'].apply(classify_score)
data.head(10)


# In[37]:


# Replacing the values in 'Grade' with Grade Mapping
grade_mapping = {
    'credit': 0,
    'merit': 1,
    'distinction': 2
}
data['Grade'] = data['Grade'].replace(grade_mapping)


# In[47]:


data


# In[57]:


# Separate features and target
X = data.drop('Grade', axis=1)
y = data['Grade']


# In[59]:


X


# In[61]:


y


# In[53]:





# In[63]:


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[65]:


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# In[77]:


scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.fit_transform(X_test)


# In[85]:


xgb = XGBClassifier(n_estimators=100, learning_rate=0.01, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(probability=True, random_state=42)

# Define the stacking ensemble with Logistic Regression as the meta-learner
base_models = [('xgb', xgb), ('rf', rf), ('svm', svm)]
stacking_ensemble = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())

# Train the stacking ensemble
stacking_ensemble.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred_stacking = stacking_ensemble.predict(X_test)

# Evaluate the model
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
print(f'Stacking Ensemble Accuracy: {accuracy_stacking:.4f}')


# In[96]:


new_X = data[['Attendance',
 'Hours_Studied',
 'Previous_Scores',
 'Motivation_Level',
 'Access_to_Resources',
 'Tutoring_Sessions',
 'Physical_Activity',
 'Distance_from_Home',
 'Family_Income',
 'Sleep_Hours']]
new_y = data['Grade']


# In[97]:


new_y


# In[98]:


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, test_size=0.2, random_state=42)


# In[99]:


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# In[100]:


scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.fit_transform(X_test)


# In[108]:


xgb = XGBClassifier(n_estimators=100, learning_rate=0.01, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(probability=True, random_state=42)

# Define the stacking ensemble with Logistic Regression as the meta-learner
base_models = [('xgb', xgb), ('rf', rf), ('svm', svm)]
stacking_ensemble_10 = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())

# Train the stacking ensemble
stacking_ensemble_10.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred_stacking = stacking_ensemble.predict(X_test)

# Evaluate the model
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
print(f'Stacking Ensemble Accuracy: {accuracy_stacking:.4f}')


# In[112]:


stacking_ensemble_10.predict([[1,3,43,4,22,2,4,2,3,4]])


# In[116]:


import pickle
# Save the model using pickle
with open('stacking_ensemble_model.pkl', 'wb') as file:
    pickle.dump(stacking_ensemble, file)


# In[114]:


# Assuming you have trained your models and meta-model

import os
import joblib  # Assuming scikit-learn < 0.24

# Specify the directory path (modify if needed)
directory = "/Users/apple/Desktop/DATA SCIENCE PROJECT PREPARARTION /DATA/"

# Create the directory if it doesn't exist (avoids errors)
os.makedirs(directory, exist_ok=True)

# Save base models (modify filenames as needed)
#joblib.dump(rf_model, os.path.join(directory, "rf_model.pkl"))
#joblib.dump(xgb_model, os.path.join(directory, "xgb_model.pkl"))
#joblib.dump(et_model, os.path.join(directory, "et_model.pkl"))

# Save meta-model
if joblib.dump(stacking_ensemble_10, os.path.join(directory, "stack.pkl")):  # Assuming meta-model is trained
    print("Ensemble Model components saved successfully!")

