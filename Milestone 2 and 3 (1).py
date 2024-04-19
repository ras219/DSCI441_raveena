#!/usr/bin/env python
# coding: utf-8

# # 1. Import Packages and Read DataFrame

# In[1]:


# for data manipulation
import pandas as pd
import numpy as np

# for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# for preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# for model training
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline


# for model evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report



# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# function for Sub-Heading
def heading(title):
    print('-'*80)
    print(title.upper())
    print('-'*80)


# In[2]:


# Assuming your dataset is stored in a CSV file named 'google_play_store.csv'
df = pd.read_csv('googleplaystore.csv')


# # 2. EDA

# In[3]:


#df['Rating'].value_counts().plot(kind='bar', figsize=(10, 6))
df['Rating'].plot(kind='hist', bins=10, edgecolor='black')


# In[4]:


# df = df[(df.Rating != 19.0)]


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt


# List of columns to analyze
columns_to_analyze = ['Category', 'Installs', 'Content Rating', 'Genres']


for column in columns_to_analyze:
    # Plot counts for the current column
    df[column].value_counts().plot(kind='bar', figsize=(10, 6))
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Count of Records per {column}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Print number of unique values in the current column
    print(f"Number of unique {column}s: {df[column].nunique()}")

    # Calculate average for the current column
    avg_by_column = df.groupby(column)['Rating'].mean()
    highest_avg_value = avg_by_column.max()
    highest_avg_value_name = avg_by_column.idxmax()
    print(f"The {column} with the highest count is '{highest_avg_value_name}' with a max mean rating of {highest_avg_value}")

    print()  # Add a line break between iterations


# In[6]:


for cols in ['Category', 'Installs', 'Content Rating', 'Genres']:
    summary_stats = df.groupby(cols)['Rating'].agg(['count', 'mean', 'median', 'min', 'max'])
    print(summary_stats)


# In[7]:


df.columns


# In[8]:


# Get value counts for the column
value_counts = df['Genres'].value_counts()

# Select the top 10 value counts
top_10_counts = value_counts.head(10)

# Plotting bar chart for top 10 value counts
top_10_counts.plot(kind='bar', figsize=(10, 6))


# In[9]:


df['Category'].nunique()


# In[10]:


df[(df.Category == '1.9')]


# In[11]:


df.dtypes


# In[12]:


# No of rows and columns in the dataset
shape = df.shape
print(f'There are {shape[0]} rows and {shape[1]} columns in the dataset.')


# In[13]:


heading('Dataset Information')
df.info() # checking information about the dataset


# In[14]:


df.describe()


# In[15]:


# Counting the number of float columns
float_columns = df.select_dtypes('float').columns.value_counts()

# Counting the number of object columns
object_columns = df.select_dtypes('object').columns.value_counts().sum()

# Printing the total number of numeric and object columns in the dataset
print(f'There are {float_columns.sum()} numeric columns and {object_columns} object columns in the dataset.')


# In[16]:


# Make a copy of the dataframe to avoid modifying the original data
df2 = df.copy()

# Convert categorical columns to numerical using label encoding
# Iterate over categorical columns and apply label encoding
cat_cols = df2.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    df2[col] = LabelEncoder().fit_transform(df2[col])

# Calculate correlation matrix
correlation_matrix = df2.corr()

# Sort correlation features
sort_corr_features = correlation_matrix.index 

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df2[sort_corr_features].corr(), annot=True, cmap='RdBu', alpha=0.9, square=False)
plt.show()


# In[17]:


print(f"The names of columns in this dataset are as follows:\n {df.columns}")


# In[18]:


print(f"The Number of Rows are {df.shape[0]} and the Number of Columns are {df.shape[1]}")


# In[19]:


avg_cat_mean = df.groupby('Category')['Rating'].mean().sort_values(ascending = False)
#avg_cat_mean
plt.figure(figsize=(12,8))
sns.barplot(x = avg_cat_mean.values, y = avg_cat_mean.index)
plt.xlabel('Avg Rating')
plt.ylabel('Category')
plt.title('Average rating by category')
plt.show()


# In[20]:


df.head()


# In[ ]:





# In[21]:


df = df[(df['Last Updated'] != '1.0.19')]


# In[22]:


df['Last Updated'] = pd.to_datetime(df['Last Updated'])
df['Year'] = df['Last Updated'].apply(lambda x: x.strftime('%Y'))
df.groupby('Year')['Rating'].mean().plot(kind='line')
plt.ylabel('Rating')
plt.grid(True)
plt.show()


# # 3. Data Cleaning - Data Types and Strings attached to non string cols

# In[23]:


df['Last Updated'] = pd.to_datetime(df['Last Updated'])
df['Year'] = df['Last Updated'].apply(lambda x: x.strftime('%Y'))


# **Size Variable**

# In[24]:


#The size column should be numerical , lets convert that 
df['Size'].value_counts()
df['Size'].loc[df['Size'].str.contains('M')].value_counts().sum()


# In[25]:


df['Size'].loc[df['Size'].str.contains('k')].value_counts().sum()


# In[26]:


df['Size'].loc[df['Size'].str.contains('Varies with device')].value_counts().sum()


# In[27]:


def convert_size_to_bytes(size_str):
    if size_str == 'Varies with device':
        return np.nan 
    elif size_str.endswith('M'):
        return float(size_str[:-1]) * 1024 * 1024  
    elif size_str.endswith('k'):
        return float(size_str[:-1]) * 1024 
    elif size_str.endswith('k'):
        return float(size_str[:-1])  
    else:
        return np.nan


# In[28]:


df['Size'] = df['Size'].apply(convert_size_to_bytes)


# In[29]:


df.rename(columns={'Size': "Size_in_bytes"}, inplace=True)


# In[30]:


df['Size_in_bytes'].value_counts()


# **Install Variable**

# In[31]:


df['Installs'] = df['Installs'].apply(lambda x : x.replace('+',"") if '+' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x : x.replace(',',"") if ',' in str(x) else x)
df['Installs'].value_counts()
#filter out the one outlier with free
df = df[(df.Installs != "Free")]
#convert to int
df['Installs'] = df['Installs'].astype(int)


# In[32]:


#Possibly use this instead of the continous variable, keep both options for now


# In[33]:


bins = [-1, 0, 10, 1000, 10000, 100000, 1000000, 10000000, 10000000000]
labels=['no', 'Very low', 'Low', 'Moderate', 'More than moderate', 'High', 'Very High', 'Top Notch']
df['Installs_category'] = pd.cut(df['Installs'], bins=bins, labels=labels)


# **Price Column**

# In[34]:


df['Price'].loc[df['Price'].str.contains('\$')].value_counts().sum()


# In[35]:


df['Price'] = df['Price'].apply(lambda x : x.replace('$',"") if '$' in str(x) else x)
df['Price'] = df['Price'].apply(lambda x: float(x))


# **Values with many categorical values (before encoding it)**

# In[36]:


# Process 'Genres' column to keep top 5 values and group the rest as 'Other'
top_genres = df['Genres'].value_counts().nlargest(5).index.tolist()
df['Genres'] = df['Genres'].apply(lambda x: x if x in top_genres else 'Other')


# In[37]:


# Process 'Genres' column to keep top 5 values and group the rest as 'Other'
top_cat = df['Category'].value_counts().nlargest(5).index.tolist()
df['Category'] = df['Category'].apply(lambda x: x if x in top_cat else 'Other')


# # 4 - Data Cleaning - Null Values

# In[38]:


heading('Percentage of missing values in each column')
# Calculate the percentage of missing values in each column
df.isnull().sum().sort_values(ascending=False) / len(df) * 100


# In[39]:


heading('Percentage of missing values in each column')
# Calculate the percentage of missing values in each column
df.isnull().sum().sort_values(ascending=False) / len(df) * 100
heading('visulalizing the missing values in the dataset')
# Set up the figure size for the plot
plt.figure(figsize=(18, 9))

# Create the heatmap to plot null values
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='RdPu')

# Show the plot
plt.show()


# In[40]:


df_non_null = df.dropna()


# In[41]:


df_cleaned_nulls = df.copy()
df_cleaned_nulls['Rating'] = df_cleaned_nulls['Rating'].fillna(df_cleaned_nulls.Rating.mean())
df_cleaned_nulls['Size_in_bytes'] = df_cleaned_nulls['Size_in_bytes'].fillna(df_cleaned_nulls.Size_in_bytes.mean())
df_cleaned_nulls = df_cleaned_nulls.dropna()


# In[42]:


# i have three versions of the dataframe now that i can try modeling with each, one with null values one 
#with non null values due to dropping and due to filling with the mean 


# In[43]:


df_non_null.shape


# In[44]:


df.shape


# # 5 - Modeling - Regression Models

# **One Hot Encode First**

# In[82]:


# Select columns for X (features) and y (target)
selected_features = ['Reviews', 'Size_in_bytes', 'Installs', 'Price','Year','Category','Genres', 'Type']
X = df_cleaned_nulls[selected_features]
y = df_cleaned_nulls['Rating']
# Perform one-hot encoding on categorical columns in X
categorical_cols = ['Category','Genres', 'Type']
X_encoded = pd.get_dummies(X, columns=categorical_cols)


# **Train Test Split**

# In[83]:


# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# **Train and Eval Models**

# In[84]:


# Initialize and train Linear Regression model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)

# Initialize and train Ridge Regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Initialize and train Lasso Regression model
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)

# Initialize and train Gradient Boosting Regressor model
gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
gb_regressor.fit(X_train, y_train)


# Predict on the test set
lin_reg_pred = lin_reg_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)
lasso_pred = lasso_model.predict(X_test)
gb_pred = gb_regressor.predict(X_test)
#svr_pred = svr_model.predict(X_test)

# Calculate evaluation metrics
models = {
    'Linear Regression': lin_reg_pred,
    'Ridge Regression': ridge_pred,
    'Lasso Regression': lasso_pred,
    'Gradient Boosting Regressor': gb_pred
}

metrics = {}
for model_name, y_pred in models.items():
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics[model_name] = {'MSE': mse, 'MAE': mae, 'R2': r2}

# Print evaluation metrics for all models
for model_name, eval_metrics in metrics.items():
    print(model_name, "Metrics:")
    print("Mean Squared Error (MSE):", eval_metrics['MSE'])
    print("Mean Absolute Error (MAE):", eval_metrics['MAE'])
    print("R-squared (R2):", eval_metrics['R2'])
    print()


# **Hyper Param Tuning on a GBM**

# In[ ]:


#The best model performance was on GBM so let's tune that now! 
#TAKES A LONG TIME TO RUN!! LOT OF PARAMETER OPTIONS I CHOSE INC A 5 FOLD CV


# In[85]:


# Initialize the Gradient Boosted Regression model
gb_model = GradientBoostingRegressor()

# Define the hyperparameters grid for Grid Search
param_grid = {
    'n_estimators': [100, 300],  # Number of boosting stages
    'learning_rate': [0.05,0.2],  # Learning rate shrinks the contribution of each tree
    'max_depth': [3, 5],  # Maximum depth of the individual trees
    'min_samples_split': [5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 4],  # Minimum number of samples required to be at a leaf node
}

# Initialize Grid Search with cross-validation
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)

# Perform Grid Search to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Best Model Hyperparameters:", best_params)
print("Best Model Mean Squared Error on Test Set:", mse)


# In[86]:


r2_score(y_test, y_pred)


# # 6 - Modeling - Classification Models - Non Sampled

# In[62]:


df_cleaned_nulls['Rating_class'] = df_cleaned_nulls['Rating'].apply(lambda x: 1 if x > 3.5 else 0)


# In[63]:


#!pip install xgboost

from xgboost import XGBClassifier


# In[64]:


# Select columns for X (features) and y (target)
selected_features = ['Reviews', 'Size_in_bytes', 'Installs', 'Price','Year','Category','Genres', 'Type']
X = df_cleaned_nulls[selected_features]
y = df_cleaned_nulls['Rating_class']
# Perform one-hot encoding on categorical columns in X
categorical_cols = ['Category','Genres', 'Type']
X_encoded = pd.get_dummies(X, columns=categorical_cols)
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[67]:


# Initialize and train Logistic Regression (One-vs-Rest) model
log_reg_model = LogisticRegression(multi_class='ovr', max_iter=1000)
log_reg_model.fit(X_train, y_train)

# Initialize and train Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)


# Initialize and train Gradient Boosting Classifier model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gb_model.fit(X_train, y_train)

# # Initialize and train XGBoost classifier
# xgb_model = XGBClassifier()
# xgb_model.fit(X_train, y_train)


# Predict on the test set
log_reg_pred = log_reg_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
#xgb_pred = xgb_model.predict(X_test)

# Calculate accuracy
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
gb_accuracy = accuracy_score(y_test, gb_pred)
#xgb_accuracy = accuracy_score(y_test, xgb_pred)

# Print accuracy for all models
print("Logistic Regression Accuracy:", log_reg_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
print("Gradient Boosting Classifier Accuracy:", gb_accuracy)
#print("XGBoost  Accuracy:", xgb_accuracy)

# Print classification report for one of the models (e.g., Logistic Regression)
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, log_reg_pred))

# Print classification report for one of the models (e.g., Logistic Regression)
print("\nClassification Report (RF):")
print(classification_report(y_test, rf_pred))

# Print classification report for one of the models (e.g., Logistic Regression)
print("\nClassification Report GBT ):")
print(classification_report(y_test, gb_pred))


# In[69]:


df_cleaned_nulls.groupby('Rating_class').count()


# # Resampling for more balances results

# In[70]:


from sklearn.utils import resample

# Class count before downsampling
class_counts = df_cleaned_nulls['Rating_class'].value_counts()
print("Class Counts Before Downsampling:")
print(class_counts)

# Separate majority and minority classes
class_majority = df_cleaned_nulls[df_cleaned_nulls['Rating_class'] == 1]
class_minority = df_cleaned_nulls[df_cleaned_nulls['Rating_class'] == 0]

# Downsample majority class
class_majority_downsampled = resample(class_majority,
                                      replace=False,  # Sample without replacement
                                      n_samples=len(class_minority),  # Match minority class size
                                      random_state=42)  # Reproducible results

# Combine minority class and downsampled majority class
data_downsampled = pd.concat([class_majority_downsampled, class_minority])

# Class count after downsampling
class_counts_downsampled = data_downsampled['Rating_class'].value_counts()
print("\nClass Counts After Downsampling:")
print(class_counts_downsampled)

# Now 'data_downsampled' contains the balanced dataset


# # 7 - Modeling - Classification Models -  Sampled

# In[71]:


# Select columns for X (features) and y (target)
selected_features = ['Reviews', 'Size_in_bytes', 'Installs', 'Price','Year','Category','Genres', 'Type']
X = data_downsampled[selected_features]
y = data_downsampled['Rating_class']
# Perform one-hot encoding on categorical columns in X
categorical_cols = ['Category','Genres', 'Type']
X_encoded = pd.get_dummies(X, columns=categorical_cols)
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[72]:


# Initialize and train Logistic Regression (One-vs-Rest) model
log_reg_model = LogisticRegression(multi_class='ovr', max_iter=1000)
log_reg_model.fit(X_train, y_train)

# Initialize and train Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)


# Initialize and train Gradient Boosting Classifier model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gb_model.fit(X_train, y_train)

# # Initialize and train XGBoost classifier
# xgb_model = XGBClassifier()
# xgb_model.fit(X_train, y_train)


# Predict on the test set
log_reg_pred = log_reg_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
#xgb_pred = xgb_model.predict(X_test)

# Calculate accuracy
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
gb_accuracy = accuracy_score(y_test, gb_pred)
#xgb_accuracy = accuracy_score(y_test, xgb_pred)

# Print accuracy for all models
print("Logistic Regression Accuracy:", log_reg_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
print("Gradient Boosting Classifier Accuracy:", gb_accuracy)
#print("XGBoost  Accuracy:", xgb_accuracy)

# Print classification report for one of the models (e.g., Logistic Regression)
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, log_reg_pred))

# Print classification report for one of the models (e.g., Logistic Regression)
print("\nClassification Report (RF):")
print(classification_report(y_test, rf_pred))

# Print classification report for one of the models (e.g., Logistic Regression)
print("\nClassification Report GBT ):")
print(classification_report(y_test, gb_pred))


# In[73]:


# 8 - Hyper Param Tuining on my RF


# In[74]:


# Initialize Random Forest classifier
rf_model = RandomForestClassifier()

# Define the hyperparameters grid for Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize Grid Search with cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)

# Perform Grid Search to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# In[87]:


best_params


# # Explainablility 

# In[75]:


# Make predictions using the best model
y_pred = best_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Best Model Accuracy:", accuracy)

# Generate classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


# In[79]:


get_ipython().system('pip install shap')

import shap


# In[80]:


# Calculate feature importance
feature_importance = best_model.feature_importances_

# Get feature names
feature_names = X_encoded.columns.tolist()

# Create a DataFrame to store feature importance
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance
print("Feature Importance:\n", importance_df)

# Initialize SHAP explainer
explainer = shap.Explainer(best_model)
shap_values = explainer.shap_values(X_test)

# Create SHAP summary plot
shap.summary_plot(shap_values, X_test, plot_type='bar')


# In[81]:


# Initialize SHAP explainer
explainer = shap.Explainer(best_model)
shap_values = explainer.shap_values(X_test)

# Create SHAP summary plot (beeswarm)
shap.summary_plot(shap_values, X_test, plot_type='beeswarm')


# In[89]:


# Assuming your Random Forest model is named 'rf_model'
feature_importance = best_model.feature_importances_

# Create a DataFrame to show feature importance
importance_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plotting feature importance
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()


# In[ ]:




