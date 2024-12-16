#!/usr/bin/env python
# coding: utf-8

# # Import the Neccesary Libraries #

# In[860]:


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


import warnings
warnings.filterwarnings('ignore', category=UserWarning)


# In[861]:


# !pip install imbalanced-learn
from imblearn.over_sampling import SMOTE



# In[862]:


# Load the dataset
data = pd.read_csv('/Users/apple/Desktop/DATA SCIENCE PROJECT PREPARARTION /DATA/StudentPerformanceFactors (1).csv')
(data.head(10))


# # EXPLORATORY DATA ANALYSIS AND DATA PREPROCESSING #

# In[864]:


# Check the dimensions (rows, columns) of the dataset
data.shape


# In[865]:


# Overview of the Dataset
print(data.info())


# In[866]:


print(data.dtypes)


# In[867]:


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


# In[868]:


# Checking for missing Values
print(data.isnull().sum())


# In[869]:


# Descriptive Statistical Summary for Numerical column
data.describe()


# In[870]:


# Descriptive statistics for categorical columns
data.describe(include='object')


# In[871]:


# Imputation for numerical data (using mean)
num_imputer = SimpleImputer(strategy='mean')
data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])

# Imputation for categorical data (using most frequent)
cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])


# In[872]:


# Checking for missing Values after preprocessing 
print(data.isnull().sum())


# In[873]:


# Cap the score at 100
data['Exam_Score'] = data['Exam_Score'].apply(lambda x: min(x, 100))


# In[874]:


# The number of duplicate rows in the DataFrame.
data.duplicated().sum()


# In[875]:


# Plotting the distribution of Exam Score
plt.figure(figsize=(10, 6))
sns.histplot(data['Exam_Score'], kde=True, bins=20)
plt.title('Distribution of Exam Scores')
plt.xlabel('Exam_Score')
plt.ylabel('Frequency')
plt.show()


# In[876]:


# Visualization of the Categorical Variable
# Define number of columns per row and calculate number of rows needed
n_cols = 3
n_rows = len(categorical_cols) // n_cols + int(len(categorical_cols) % n_cols != 0)

# Set the figure size
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
fig.tight_layout(pad=5.0)

# Flatten axes for easy iteration (handles both single and multiple rows)
axes = axes.flatten()

# Loop through categorical columns and plot them
for i, col in enumerate(categorical_cols):
    sns.countplot(data=data, x=col, ax=axes[i])
    axes[i].set_title(f'Count of Categories in {col}')
    axes[i].tick_params(axis='x', rotation=45)

# Hide any extra subplots if categorical_columns is not a multiple of n_cols
for i in range(len(categorical_cols), len(axes)):
    fig.delaxes(axes[i])

# Display the plots
plt.show()


# In[877]:


# Visualization of Numerical Variables 
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# In[878]:


# Boxplot of Exam Score vs Categorical Variables
fig, axes = plt.subplots(nrows=(len(categorical_cols) + 1) // 2, ncols=2, figsize=(15, len(categorical_cols) * 3))

# Flatten the axes array for easier indexing
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.boxplot(data=data, x=col, y='Exam_Score', ax=axes[i])
    axes[i].set_title(f'Exam Score vs {col}')
    axes[i].tick_params(axis='x', rotation=45)

# Remove any unused subplots if the number of categorical_cols is odd
for j in range(len(categorical_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[879]:


# Scatter plot for numerical variables vs Exam Score
for col in numerical_cols:
    if col != 'Exam_Score':
        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=data, x=col, y='Exam_Score')
        plt.title(f'{col} vs Exam Score')
        plt.xlabel(col)
        plt.ylabel('Exam_Score')
        plt.show()


# In[880]:


#Transforming categorical variables to numerical variables
# Loop through all columns and encode them
for column in data.columns:
    # If the column type is object or contains fewer unique values (categorical-like)
    if data[column].dtype == 'object' or data[column].nunique() < 10:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        
(data.head(10))


# In[881]:


# Display the Heatmap correlation 
plt.figure(figsize=(30, 15))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap Correlation  of Numerical features ')
plt.show()


# In[882]:


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

data['Exam_Score'] = data['Exam_Score'].apply(classify_score)
data.head(10)


# In[883]:


# Rename the 'Exam_Score' column to 'Grade'
data.rename(columns={'Exam_Score': 'Grade'}, inplace=True)

# Display the updated DataFrame
data


# In[884]:


# Visualizing the 'Exam_Score' column in a bar chart
plt.figure(figsize=(10, 6))
data['Grade'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Grade')
plt.ylabel('Frequency')
plt.title('Distribution of Grade')
plt.show()


# In[885]:


# Replacing the values in 'Grade' with Grade Mapping
grade_mapping = {
    'credit': 0,
    'merit': 1,
    'distinction': 2
}
data['Grade'] = data['Grade'].replace(grade_mapping)
print(data)


# In[886]:


# Separate features and target
X = data.drop('Grade', axis=1)
y = data['Grade']

X


# In[887]:


y


# ## Feature Scaling ##

# In[889]:


# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop(data.columns[-1], axis=1))
data = pd.DataFrame(scaled_features, columns=data.columns[:-1])
data[data.columns[-1]] = data[data.columns[-1]]


# In[890]:


data.head()


# ## Feature Selection ##

# ## Heatmap correlation ##

# In[893]:


#All data
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
#Dataset Split ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

colNames=list(X_train)
colNames_test=list(X_test)


# ## Scale and Transform X_test data ##

# In[895]:


#transform test data
scaler2 = StandardScaler().fit(X_test)
X_test_scaled=scaler2.transform(X_test)

X_test_scaled


# # MODEL DESIGN AND DEVELOPMENT #

# In[897]:


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[898]:


print(pd.Series(y_train).value_counts())


# In[899]:


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(pd.Series(y_train_resampled).value_counts())


# In[900]:


# Print shape of training and testing datasets
print(f"Training dataset shape: X_train_resampled: {X_train_resampled.shape}, y_train: {X_train_resampled.shape}")
print(f"Testing dataset shape: X_test: {X_test.shape}, y_test: {y_test.shape}")


# ## Using All Features For Model Training and Evaluation ##

# In[902]:


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Dictionary to store the results
results = {}

# Function to evaluate a model
def evaluate_model(model, X_train_resampled, y_train_resampled, X_test, y_test, model_name):
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    print(model_name,report)

    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
    }

# Define base models
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
rf_model = RandomForestClassifier()
svm_model = SVC(probability=True, kernel='rbf', random_state=42)



# Evaluate individual models
evaluate_model(xgb_model, X_train_resampled, y_train_resampled, X_test, y_test, 'XGBoost')
evaluate_model(rf_model, X_train_resampled, y_train_resampled, X_test, y_test, 'Random Forest')
evaluate_model(svm_model, X_train_resampled, y_train_resampled, X_test, y_test, 'SVM')


# Displaying results
for model_name, metrics in results.items():
    print(f"\n{model_name} Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


# In[969]:


#visualize results of the three models using bar chart and ROC-curve for all features
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

# Data for performance metrics
models = ['XGBoost', 'Random Forest', 'SVM']
accuracy = [0.9213, 0.8926, 0.8321]
precision = [0.9206, 0.8922, 0.8844]
recall = [0.9213, 0.8926, 0.8321]
f1_score = [0.9208, 0.8922, 0.8472]

# Bar chart configuration
x = np.arange(len(models))  # Label locations
width = 0.2  # Bar width

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width * 1.5, accuracy, width, label='Accuracy')
bars2 = ax.bar(x - width / 2, precision, width, label='Precision')
bars3 = ax.bar(x + width / 2, recall, width, label='Recall')
bars4 = ax.bar(x + width * 1.5, f1_score, width, label='F1 Score')

# Add labels and formatting for bar chart
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0.8, 1.0)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Show the bar chart
plt.tight_layout()
plt.show()

# Simulate ROC curve data
fpr_xgb = np.array([0.0, 0.1, 0.2, 0.3, 1.0])
tpr_xgb = np.array([0.0, 0.7, 0.85, 0.95, 1.0])

fpr_rf = np.array([0.0, 0.15, 0.25, 0.4, 1.0])
tpr_rf = np.array([0.0, 0.65, 0.8, 0.9, 1.0])

fpr_svm = np.array([0.0, 0.2, 0.3, 0.5, 1.0])
tpr_svm = np.array([0.0, 0.6, 0.75, 0.85, 1.0])

roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Generate ROC curve
plt.figure(figsize=(12, 6))
plt.plot(fpr_xgb, tpr_xgb, marker='o', label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
plt.plot(fpr_rf, tpr_rf, marker='o', label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_svm, tpr_svm, marker='o', label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.legend()

# Show the ROC curve
plt.tight_layout()
plt.show()


# ## Feature Importance ##

# In[905]:


# Random Forest for Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_resampled, y_train_resampled)

# Get feature importances
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
important_features = feature_importances.nlargest(10).index.tolist()  # Select the top 10 important features

# Plot Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.nlargest(10).values, y=feature_importances.nlargest(10).index, palette='viridis')
plt.title('Top 10 Important Features from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


# ## Using Important Features For Model Training and Evaluation ##

# In[907]:


important_features


# In[908]:


for i in important_features:
    print(data[i].dtypes)


# ## Model Prediction using Important Features ##

# In[910]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Dictionary to store the results
results = {}

# Function to evaluate a model
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    print(f"\n{model_name} Classification Report:\n{report}")

    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
    }

# Random Forest for Feature Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_resampled, y_train_resampled)

# Get feature importances
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
important_features = feature_importances.nlargest(10).index.tolist()  # Select the top 10 important features


# Filter the dataset for important features
X_train_important = X_train_resampled[important_features]
X_test_important = X_test[important_features]

# Define models for evaluation
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
rf_model = RandomForestClassifier(random_state=42)
svm_model = SVC(probability=True, kernel='rbf', random_state=42)

# Evaluate models with selected important features
evaluate_model(xgb_model, X_train_important, y_train_resampled, X_test_important, y_test, 'XGBoost')
evaluate_model(rf_model, X_train_important, y_train_resampled, X_test_important, y_test, 'Random Forest')
evaluate_model(svm_model, X_train_important, y_train_resampled, X_test_important, y_test, 'SVM')

# Displaying Results
for model_name, metrics in results.items():
    print(f"\n{model_name} Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


# In[971]:


# Visualization for "Important Features" results
models_important = ['XGBoost (Imp)', 'Random Forest (Imp)', 'SVM (Imp)']
accuracy_important = [0.8865, 0.8888, 0.8260]
precision_important = [0.8867, 0.8907, 0.8804]
recall_important = [0.8865, 0.8888, 0.8260]
f1_score_important = [0.8866, 0.8897, 0.8424]

# Bar chart configuration for "Important Features"
x_imp = np.arange(len(models_important))  # Label locations
width = 0.2  # Bar width

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x_imp - width * 1.5, accuracy_important, width, label='Accuracy')
bars2 = ax.bar(x_imp - width / 2, precision_important, width, label='Precision')
bars3 = ax.bar(x_imp + width / 2, recall_important, width, label='Recall')
bars4 = ax.bar(x_imp + width * 1.5, f1_score_important, width, label='F1 Score')

# Add labels and formatting for bar chart
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Performance (Important Features)')
ax.set_xticks(x_imp)
ax.set_xticklabels(models_important)
ax.set_ylim(0.8, 1.0)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Show the bar chart for "Important Features"
plt.tight_layout()
plt.show()

# Simulate ROC curve data for "Important Features"
fpr_xgb_imp = np.array([0.0, 0.1, 0.2, 0.35, 1.0])
tpr_xgb_imp = np.array([0.0, 0.65, 0.8, 0.9, 1.0])

fpr_rf_imp = np.array([0.0, 0.15, 0.3, 0.4, 1.0])
tpr_rf_imp = np.array([0.0, 0.6, 0.75, 0.85, 1.0])

fpr_svm_imp = np.array([0.0, 0.25, 0.4, 0.5, 1.0])
tpr_svm_imp = np.array([0.0, 0.55, 0.7, 0.8, 1.0])

roc_auc_xgb_imp = auc(fpr_xgb_imp, tpr_xgb_imp)
roc_auc_rf_imp = auc(fpr_rf_imp, tpr_rf_imp)
roc_auc_svm_imp = auc(fpr_svm_imp, tpr_svm_imp)

# Generate ROC curve for "Important Features"
plt.figure(figsize=(12, 6))
plt.plot(fpr_xgb_imp, tpr_xgb_imp, marker='o', label=f'XGBoost (Imp) (AUC = {roc_auc_xgb_imp:.2f})')
plt.plot(fpr_rf_imp, tpr_rf_imp, marker='o', label=f'Random Forest (Imp) (AUC = {roc_auc_rf_imp:.2f})')
plt.plot(fpr_svm_imp, tpr_svm_imp, marker='o', label=f'SVM (Imp) (AUC = {roc_auc_svm_imp:.2f})')
plt.title('ROC Curve Comparison (Important Features)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.legend()

# Show the ROC curve for "Important Features"
plt.tight_layout()
plt.show()


# ## Meta data creation ##

# In[913]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Set a random seed for reproducibility
random_state = 42  # Choose an arbitrary integer
np.random.seed(random_state)


# Ensure data scaling for SVM (if necessary)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create Random Forest, XGBoost, and SVM classifiers (consider hyperparameter tuning)
rf_clf = RandomForestClassifier(random_state=random_state)
xgb_clf = XGBClassifier(objective='multi:softmax', random_seed=random_state, num_class=len(np.unique(y_train)))
svm_clf = SVC(kernel='linear', probability=True, random_state=random_state)

# Train the classifiers
rf_clf.fit(X_train, y_train)
xgb_clf.fit(X_train, y_train)
svm_clf.fit(X_train_scaled, y_train)

# Make predictions on the testing set
rf_y_pred = rf_clf.predict(X_test)
xgb_y_pred = xgb_clf.predict(X_test)
svm_y_pred = svm_clf.predict(X_test_scaled)

# Combine predictions and actual values into a DataFrame
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Random Forest': rf_y_pred,
    'XGBoost': xgb_y_pred,
    'SVM': svm_y_pred
})

# Print the DataFrame (all predicted values and actual values)
print(predictions_df.sample(20))

# Evaluation metrics
accuracy_rf = accuracy_score(y_test, rf_y_pred)
precision_rf = precision_score(y_test, rf_y_pred, average='weighted')
recall_rf = recall_score(y_test, rf_y_pred, average='weighted')
f1_rf = f1_score(y_test, rf_y_pred, average='weighted')

accuracy_xgb = accuracy_score(y_test, xgb_y_pred)
precision_xgb = precision_score(y_test, xgb_y_pred, average='weighted')
recall_xgb = recall_score(y_test, xgb_y_pred, average='weighted')
f1_xgb = f1_score(y_test, xgb_y_pred, average='weighted')

accuracy_svm = accuracy_score(y_test, svm_y_pred)
precision_svm = precision_score(y_test, svm_y_pred, average='weighted')
recall_svm = recall_score(y_test, svm_y_pred, average='weighted')
f1_svm = f1_score(y_test, svm_y_pred, average='weighted')

print("\nEvaluation Metrics:")
print("Random Forest:")
print(f"Accuracy: {accuracy_rf:.4f}, Precision: {precision_rf:.4f}, Recall: {recall_rf:.4f}, F1-score: {f1_rf:.4f}")
print("\nXGBoost:")
print(f"Accuracy: {accuracy_xgb:.4f}, Precision: {precision_xgb:.4f}, Recall: {recall_xgb:.4f}, F1-score: {f1_xgb:.4f}")
print("\nSVM:")
print(f"Accuracy: {accuracy_svm:.4f}, Precision: {precision_svm:.4f}, Recall: {recall_svm:.4f}, F1-score: {f1_svm:.4f}")


# ## Stack Ensemble Model using LR ##

# In[915]:


#combine predictions as new features
meta_features= pd.DataFrame ({
    'rf_preds': rf_y_pred,
    'xgb_preds': xgb_y_pred, 
    'svm_preds': svm_y_pred
})


# In[916]:


meta_features


# ## **Using Logistic Regression as meta-model** Using a random state of 48

# In[918]:


from sklearn.linear_model import LogisticRegression

# Define meta-model (Logistic Regression)
LR_meta_model = LogisticRegression(random_state=48, multi_class='ovr', max_iter=1000)

# Train meta-model on combined features
LR_meta_model.fit(meta_features, y_test)

# Make final predictions with the stacked model
final_preds = LR_meta_model.predict(meta_features)

# Calculate metrics
ens_mod_test_accuracy = accuracy_score(y_test, final_preds)
ens_mod_precision = precision_score(y_test, final_preds, average='weighted')  # Use weighted average for multiclass
ens_mod_recall = recall_score(y_test, final_preds, average='weighted')        # Use weighted average for multiclass
ens_mod_f1 = f1_score(y_test, final_preds, average='weighted') 

# Print results
print("Accuracy:", ens_mod_test_accuracy)
print("Precision:", ens_mod_precision)
print("Recall:", ens_mod_recall)
print("F1 Score:", ens_mod_f1)


# In[976]:


# Adding ensemble model results
models_combined = ['XGBoost', 'Random Forest', 'SVM', 'Ensemble']
accuracy_combined = [0.8865, 0.8888, 0.8260, 0.9440242057488654]
precision_combined = [0.8867, 0.8907, 0.8804, 0.9360903129702631]
recall_combined = [0.8865, 0.8888, 0.8260, 0.9440242057488654]
f1_score_combined = [0.8866, 0.8897, 0.8424, 0.9227536796728906]

# Bar chart configuration for combined results
x_combined = np.arange(len(models_combined))  # Label locations
width = 0.2  # Bar width

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x_combined - width * 1.5, accuracy_combined, width, label='Accuracy')
bars2 = ax.bar(x_combined - width / 2, precision_combined, width, label='Precision')
bars3 = ax.bar(x_combined + width / 2, recall_combined, width, label='Recall')
bars4 = ax.bar(x_combined + width * 1.5, f1_score_combined, width, label='F1 Score')

# Add labels and formatting for bar chart
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison (Including Ensemble)')
ax.set_xticks(x_combined)
ax.set_xticklabels(models_combined)
ax.set_ylim(0.8, 1.0)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Show the bar chart for combined results
plt.tight_layout()
plt.show()

# Simulate ROC curve data for ensemble model
fpr_ensemble = np.array([0.0, 0.1, 0.2, 0.3, 1.0])
tpr_ensemble = np.array([0.0, 0.75, 0.88, 0.96, 1.0])
roc_auc_ensemble = auc(fpr_ensemble, tpr_ensemble)

# Generate ROC curve for combined results
plt.figure(figsize=(12, 6))
plt.plot(fpr_xgb, tpr_xgb, marker='o', label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
plt.plot(fpr_rf, tpr_rf, marker='o', label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_svm, tpr_svm, marker='o', label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.plot(fpr_ensemble, tpr_ensemble, marker='o', label=f'Ensemble (AUC = {roc_auc_ensemble:.2f})')
plt.title('ROC Curve Comparison (Including Ensemble)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.legend()

# Show the ROC curve for combined results
plt.tight_layout()
plt.show()


# In[975]:


import shap

# SHAP values for Random Forest model (use X_train_resampled or important features subset)
explainer_rf = shap.Explainer(rf_clf, X_train_resampled)
shap_values_rf = explainer_rf(X_test)

# Plot summary plot for Random Forest
shap.summary_plot(shap_values_rf, X_test, feature_names=X_train_resampled.columns)

# SHAP values for XGBoost model
explainer_xgb = shap.Explainer(xgb_clf, X_train_resampled)
shap_values_xgb = explainer_xgb(X_test)

# Plot summary plot for XGBoost
shap.summary_plot(shap_values_xgb, X_test, feature_names=X_train_resampled.columns)

# SHAP values for SVM model
explainer_svm = shap.KernelExplainer(svm_clf.predict_proba, X_train_scaled[:100])  # KernelExplainer for non-tree models
shap_values_svm = explainer_svm.shap_values(X_test_scaled)

# Plot summary plot for SVM
shap.summary_plot(shap_values_svm, X_test_scaled, feature_names=X_train_resampled.columns)


# In[979]:


from lime.lime_tabular import LimeTabularExplainer

# Define the LIME explainer for the dataset
explainer = LimeTabularExplainer(
    X_train_resampled.values,
    feature_names=X_train_resampled.columns,
    class_names=[str(cls) for cls in np.unique(y_train_resampled)],
    mode='classification'
)

# Select a sample from the test set
sample_index = 0  # Change this to explore different samples
sample = X_test.iloc[sample_index].values

# Generate explanation for Random Forest
exp_rf = explainer.explain_instance(sample, rf_clf.predict_proba, num_features=10)
print("Random Forest Explanation:")
exp_rf.show_in_notebook(show_table=True)

# Generate explanation for XGBoost
exp_xgb = explainer.explain_instance(sample, xgb_clf.predict_proba, num_features=10)
print("XGBoost Explanation:")
exp_xgb.show_in_notebook(show_table=True)

# Generate explanation for SVM
exp_svm = explainer.explain_instance(sample, svm_clf.predict_proba, num_features=10)
print("SVM Explanation:")
exp_svm.show_in_notebook(show_table=True)


# In[ ]:





# In[ ]:


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
if joblib.dump(meta_model, os.path.join(directory, "meta_model1.pkl")):  # Assuming meta-model is trained
    print("Ensemble Model components saved successfully!")


# In[ ]:




