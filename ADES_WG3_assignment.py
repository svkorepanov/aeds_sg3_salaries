#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[1]:


import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, learning_curve
from sklearn import metrics
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import dtreeviz

# https://www.scikit-yb.org/
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# Avoiding warnings to pop up
import warnings
warnings.filterwarnings("ignore")

# Set aspects of the visual theme for all matplotlib and seaborn plots.
sns.set()

os.getcwd()


# # Understanding the Dataset
# 
# Let's look at the first few rows of the dataset
# Even from this small sample we can see a lot of missing data and data that is not relevant.
# 
# For example some of the job postings are missing `salary data` which is very important for our project.
# 

# In[2]:


df = pd.read_csv('./content/sample_data/us-software-engineer-jobs-zenrows.csv')
df.head()


# ## Information about data set
# 
# From the information we are able to see all type of data in our disposal.
# Let list a few of them that are more important to us and describe their meaning
# 
# 
# 
# *   **Title** - title of the job position on which company is hiring
# *   **Company** - name of the company that is hiring
# *   **Salary** - Salary range that the company is offering for the position
# *   **Location** - Location where the office of the company is located
# *   **Types** - Type of employment: Full-time, Part-time, Contract, Internship, Temporary
# 
# Other data points are considered unrelated for our goals. Therefore, let's remove them.
# 
# 

# In[3]:


df.info()
columns_to_drop = ['rating', 'review_count', 'relative_time', 'hires_needed', 'hires_needed_exact',
                   'remote_work_model', 'snippet', 'dradis_job', 'link', 'new_job',
                   'job_link', 'featured_employer', 'indeed_applyable', 'ad_id', 'remote_location',
                   'source_id', 'hiring_event_job', 'indeed_apply_enabled', 'job_location_postal', 'company_overview_link',
                   'activity_date', 'location_extras']
df_cleaned = df.drop(columns_to_drop, axis=1)


# ## Cleaned Data Set
# Here is the data remained after cleaning

# In[4]:


df_cleaned.head()


# ## Normalising Data
# After we cleaned irrelevant data we need to normalise salary ranges. In the code snippet bellow we can see that salary range for a position can be stated as "a year", "a month", "a week", "an hour"
# 
# - First of all we will convert salary data to dollars per year.
# 
# - And after we will split range data to min_salary and max_salary for the role

# ### Converting Salary Data
# let's look at the unique values in the 'salary' column to understand the data better

# In[5]:


# Get unique values after dropping NaNs
unique_salaries = set(df_cleaned['salary'].unique())
unique_salaries_df = pd.DataFrame(unique_salaries)
unique_salaries_df.head(n=100)


# ### Converting to Yearly Salary
# From previous snippet we see that the salary appears in different formats and rates. 
# - a year, a month, a week, an hour
# - range and fixed rate
# 
# - Let's try to convert all of them to yearly salary

# In[6]:


# Cleaning symbols
def clean_text(text: str):
    if pd.isna(text):
        return "NaN"
    # Split the text by ' - ' to separate the salary range
    cleaned_text = (text
                    .lower()
                    .replace('$', '')
                    .replace(',', '')
                    .replace(' - ', ' '))

    return cleaned_text

df_cleaned['salary_temp'] = df_cleaned['salary'].apply(clean_text)

df_cleaned['salary_temp']


# In[7]:


# Define the assumption for hours worked per year
hours_per_year = 1892  # Statistical data from clockify 

def convert_range_match(match):
    if match.group(3) == 'a year':
        yearly_min = float(match.group(1))
        yearly_max = float(match.group(2))
    if match.group(3) == 'a month':
        yearly_min = float(match.group(1)) * 12
        yearly_max = float(match.group(2)) * 12
    if match.group(3) == 'a week':
        yearly_min = float(match.group(1)) * 52
        yearly_max = float(match.group(2)) * 52
    if match.group(3) == 'a day':
        yearly_min = float(match.group(1)) * 260
        yearly_max = float(match.group(2)) * 260    
    if match.group(3) == 'an hour':
        yearly_min = float(match.group(1)) * hours_per_year
        yearly_max = float(match.group(2)) * hours_per_year        
    return f"{yearly_min:.0f} {yearly_max:.0f}"

def convert_fixed_match(match):
    if match.group(2) == 'a year':
        yearly_min = float(match.group(1))
        yearly_max = yearly_min
    if match.group(2) == 'a month':
        yearly_min = float(match.group(1)) * 12
        yearly_max = yearly_min
    if match.group(2) == 'a week':
        yearly_min = float(match.group(1)) * 52
        yearly_max = yearly_min
    if match.group(2) == 'a day':
        yearly_min = float(match.group(1)) * 260
        yearly_max = yearly_min    
    if match.group(2) == 'an hour':
        yearly_min = float(match.group(1)) * hours_per_year
        yearly_max = yearly_min        
    return f"{yearly_min:.0f} {yearly_max:.0f}"

def converter(salary):
    if salary == 'NaN':
        return salary
    
    range_pattern = r'(\d+\.?\d*) (\d+\.?\d*) (a year|a month|an hour|a week|a day)'
    fixed_pattern = r'(\d+\.?\d*) (a year|a month|an hour|a week|a day)'
    
    range_match = re.search(range_pattern, salary)
    fixed_match = re.search(fixed_pattern, salary)
    
    if range_match:
        return convert_range_match(range_match)

    if fixed_match:
        return convert_fixed_match(fixed_match)

# Apply the function to the 'salary' column
df_cleaned['salary_temp'] = df_cleaned['salary_temp'].apply(converter)
salary_split = df_cleaned['salary_temp'].str.split(expand=True)

# Assigning the split parts to 'min_salary' and 'max_salary' columns
df_cleaned['min_salary'] = salary_split[0]
df_cleaned['max_salary'] = salary_split[1]

df_cleaned.drop(columns='salary_temp', inplace=True)
dataframe = df_cleaned.dropna(subset=["salary"])

dataframe.loc[:, 'min_salary'] = dataframe['min_salary'].astype(int)
dataframe.loc[:, 'max_salary'] = dataframe['max_salary'].astype(int)
dataframe['avg_salary'] = dataframe[['min_salary', 'max_salary']].mean(axis=1).astype(int)

print(dataframe)


# In[8]:


# Get the numeric columns
numeric_columns = dataframe.select_dtypes(include=[np.number]).columns

# Scale only the numeric columns
scaler = StandardScaler()
df_scaled_numeric = scaler.fit_transform(dataframe[numeric_columns])

# Convert the scaled array back to a DataFrame
df_scaled_numeric = pd.DataFrame(df_scaled_numeric, columns=numeric_columns)

# Now, concatenate the scaled numeric columns with the non-numeric columns
dataframe_scaled = pd.concat([dataframe.drop(columns=numeric_columns), df_scaled_numeric], axis=1)

# Before Normalization
plt.figure(figsize=(10, 6))
sns.barplot(x='types', y='avg_salary', data=dataframe)
plt.title('Distribution of Average Salaries Across Contract Types (Before Normalization)')
plt.xlabel('Contract Type')
plt.ylabel('Average Salary')
plt.xticks(rotation=45)
plt.show()

# Normalizing contract types (one-hot encoding)
contract_types_encoded = pd.get_dummies(dataframe_scaled['types'], prefix='contract')
dataframe_scaled = pd.concat([dataframe_scaled, contract_types_encoded], axis=1)

# Splitting the data into predictors and target variable
X = dataframe_scaled.drop(columns=['avg_salary'])  # predictors
y = dataframe_scaled['avg_salary']  # target variable

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# After Normalization
plt.figure(figsize=(10, 6))
sns.barplot(x='types', y='avg_salary', data=dataframe_scaled)
plt.title('Distribution of Average Salaries Across Contract Types (After Normalization)')
plt.xlabel('Contract Type')
plt.ylabel('Average Salary')
plt.xticks(rotation=45)
plt.show()


# ### Extracting State from Location

# In[9]:


# Function to extract state abbreviation from location
def extract_state(location):
    if location.lower() == 'remote':
        return 'Remote'
    else:
        match = re.search(r',\s*([A-Z]{2})$', location)
        if match:
            return match.group(1)
        else:
            return 'Unknown'

# Apply the function to create a new column 'state'

dataframe.loc[:,'state'] = dataframe.loc[:,'location'].apply(extract_state)


# ## Descriptive analysis

# In[10]:


dataframe.describe()


# ### Most frequent numbers

# In[11]:


dataframe.mode()


# ### Variance

# In[12]:


dataframe.var(numeric_only=True)


# ### Standard deviation

# In[13]:


dataframe.std(numeric_only=True)


# In[13]:





# ### Range

# In[14]:


# Calculate the range for each numeric column

column_ranges = dataframe.select_dtypes(include=[np.number]).apply(lambda x: x.max() - x.min())


# ### Interquartile Range

# In[15]:


# Calculate the interquartile range for each numeric column

column_iqr = dataframe.describe().loc['75%'] - dataframe.describe().loc['25%']


# ### Distribution analysis

# In[16]:


plt.figure(figsize=(8, 6))
plt.hist(dataframe['avg_salary'], bins=10, color='red', edgecolor='black', stacked=True)
plt.title('Distribution of Salaries')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.grid(True)
plt.show() 


# ### Frequency table

# In[17]:


# Frequency table of salaries
avg_salaries = pd.cut(dataframe['avg_salary'], bins=10, precision=0)
frequency_table = pd.Series(avg_salaries).value_counts().sort_index()
print("Avg Salaries frequency Table:")
print(frequency_table)



# ### Mean Salaries by State

# In[18]:


mean_salaries = dataframe.groupby('state')['avg_salary'].mean()

# Create a bar chart
plt.figure(figsize=(20, 6))
plt.bar(mean_salaries.index, mean_salaries.values)

plt.xlabel('State')
plt.ylabel('Mean Salary')
plt.title('Mean Salary by State')
plt.xticks(rotation=90) 

plt.show()


# ### Mean Salary by contract

# In[19]:


mean_salaries = dataframe.groupby('types')['avg_salary'].mean()

# Create a bar chart
plt.figure(figsize=(20, 6))
plt.bar(mean_salaries.index, mean_salaries.values)

plt.xlabel('Contract Type')
plt.ylabel('Mean Salary')
plt.title('Mean Salary by Contract Type')
plt.xticks(rotation=90) 

plt.show()


# ### Mean Salary by urgency

# In[20]:


mean_salaries = dataframe.groupby('urgently_hiring')['avg_salary'].mean()

# Create a bar chart
plt.figure(figsize=(20, 6))
plt.bar(mean_salaries.index, mean_salaries.values)

plt.xlabel('Contract Type')
plt.ylabel('Mean Salary')
plt.title('Mean Salary by Urgency')
plt.xticks(rotation=90) 

plt.show()


# ### Mean Salary by sponsorship

# In[21]:


mean_salaries = dataframe.groupby('sponsored')['avg_salary'].mean()

# Create a bar chart
plt.figure(figsize=(20, 6))
plt.bar(mean_salaries.index, mean_salaries.values)

plt.xlabel('Contract Type')
plt.ylabel('Mean Salary')
plt.title('Mean Salary by Ads')
plt.xticks(rotation=90) 

plt.show()


# ### Box plot of salaries by location

# In[22]:


plt.figure(figsize=(20, 6))
sns.boxplot(x='state', y='avg_salary', data=dataframe, palette='pastel', hue='state')
plt.title('Box Plot of Salaries by State')
plt.xlabel('State')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


# ### Correlation coefficient

# In[23]:


# location_mapping = {location: i for i, location in enumerate(dataframe['state'].unique())}
dataframe['state_encoded'] = dataframe['state'].astype('category').cat.codes

# Compute correlation coefficient
correlation_coefficient = dataframe['avg_salary'].corr(dataframe['state_encoded'])

print("correlation Coefficient:", correlation_coefficient)


# # Clustering

# In[24]:


dataframe['contract_encoded'] = dataframe['types'].astype('category').cat.codes
dataframe['state_encoded'] = dataframe['state'].astype('category').cat.codes
dataframe['urgency_encoded'] = dataframe['urgently_hiring'].astype('category').cat.codes
dataframe['ads_encoded'] = dataframe['sponsored'].astype('category').cat.codes

dataframe_num = dataframe[['avg_salary', 'state_encoded', 'urgency_encoded', 'ads_encoded', 'contract_encoded']]

dataframe_num


# In[25]:


scaler = StandardScaler()
df_scaled = scaler.fit_transform(dataframe_num) #generates an array
df_salaries_scaled = pd.DataFrame(df_scaled) #copies into a dataframe
df_salaries_scaled.describe()


# ### K-means

# In[26]:


k = 3
km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0) #setup the k-means algorithm


# In[27]:


km.fit(df_salaries_scaled)

print(km.cluster_centers_) # shows the centroids
print(km.labels_)          # shows the labels to which each point belongs


# In[28]:


def visualize_clusters(data, cluster_out, clust_alg=""):
    # Reduce dimensionality of data using PCA
    pca = PCA()
    pca.fit(data)
    data_pca = pca.transform(data)

    #centers_pca = pca.transform(cluster_out.cluster_centers_)
    sns.scatterplot(x=data_pca[:,0], y=data_pca[:,1],
                    hue=cluster_out.labels_,
                   palette="deep",alpha=0.5)
    #sns.scatterplot(centers_pca[:, 0], centers_pca[:, 1])

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(clust_alg + ' Clustering with PCA visualization')
    plt.show()


# In[29]:


visualize_clusters(df_salaries_scaled, km, "Kmeans")


# In[30]:


print("Avg silhouette score = ", silhouette_score(df_salaries_scaled, km.labels_))


# In[31]:


visualizer = SilhouetteVisualizer(km, colors='yellowbrick')
visualizer.fit(df_salaries_scaled) # fits the data to the visualiser
visualizer.show()                      # renders the silhouette plot


# In[32]:


visualizer = KElbowVisualizer(km, k=(1, 20))
visualizer.fit(df_salaries_scaled)
visualizer.show()


# In[33]:


k = 6
km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)  #setup the k-means algorithm
km.fit(df_salaries_scaled)

print(km.cluster_centers_)  # shows the centroids
print(km.labels_) 


# In[34]:


visualize_clusters(df_salaries_scaled, km, "Kmeans")


# In[35]:


print("Avg silhouette score = ", silhouette_score(df_salaries_scaled, km.labels_), "\n")

visualizer = SilhouetteVisualizer(km, colors='yellowbrick')
visualizer.fit(df_salaries_scaled) # fits the data to the visualiser
visualizer.show()   


# ### DBSCAN

# In[36]:


#dbscan = DBSCAN() # eps = 0.5, min_samples = 5 (default)
dbscan = DBSCAN(eps=0.5, min_samples=50) # changing the default parameters
dbscan.fit(df_salaries_scaled)


# In[37]:


print("Avg silhouette score = ", silhouette_score(df_salaries_scaled,dbscan.labels_))
print(dbscan.labels_)


# In[38]:


visualize_clusters(df_salaries_scaled, dbscan, "DBSCAN")


# In[39]:


dbscan = DBSCAN(eps=0.9, min_samples=150) # changing the default parameters
dbscan.fit(df_salaries_scaled)

print("Avg silhouette score = ", silhouette_score(df_salaries_scaled,dbscan.labels_))
print(dbscan.labels_)


# **The default values eps and min_samples suite quite well for this dataset**

# ### Agglomerative Hierarchical Clustering

# In[40]:


hac=AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='single')


# In[41]:


hac.fit(df_salaries_scaled)


# In[42]:


print("Avg silhouette score = ", silhouette_score(df_salaries_scaled,hac.labels_))


# In[43]:


visualize_clusters(df_salaries_scaled, hac, "HAC")


# # Linear Regression
# We will split the data set to training and testing first (30% testing) 
# - predictors - 'state_encoded', 'urgency_encoded', 'ads_encoded'
# - target value - 'avg_salary'

# In[44]:


cols = ['state_encoded', 'urgency_encoded', 'ads_encoded', 'contract_encoded'];

X = dataframe_num[cols]
y = dataframe_num['avg_salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[45]:


# define a function for evaluation metrics
def evaluation_metrics(pred_value):
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, pred_value))
    
    # Print out the mean squared error (mse)
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred_value))
    
    # Print out the root mean squared error (rmse)
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, pred_value)))
    
    # Print out the R-squared (R²)
    print('R-squared:', metrics.r2_score(y_test, pred_value))


# In[46]:


def residuals_plot(pred):
    residuals = y_test - pred
    plt.figure(figsize=(10, 6))
    plt.scatter(pred, residuals, alpha=0.5)
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted values")
    plt.show()


# ### Linear Regression

# In[47]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[48]:


# running predictions over a test set

y_pred = model.predict(X_test)

y_pred


# ### Evaluation Metrics (Linear Regression)

# 

# In[49]:


errors = abs(y_pred - y_test)

errors


# In[50]:


evaluation_metrics(y_pred)


# ### Visualisation

# In[51]:


residuals_plot(y_pred)


# ### Linear Regression using Gradient Descent
# Here we see R-squared really close to 0, which is awful.
# Let's try to improve performance a bit

# In[52]:


# gradient descent model 
gd_model = SGDRegressor(max_iter=30000)

gd_model.fit(X_train, y_train)


# In[53]:


gd_pred = gd_model.predict(X_test)

gd_pred


# ### Evaluation Metrics (Gradient Descent)

# In[54]:


gd_errors = abs(gd_pred - y_test)

gd_errors


# In[55]:


evaluation_metrics(gd_pred)


# It is a bit odd that R-square is negative (with 1k iterations)
# 
# But MEA and RMSE are a little worse than regular regression
# 
# Applying 30k iteration gave us similar result compared to simple Linear Regression

# In[56]:


residuals_plot(gd_pred)


# # Regression Trees

# ### Decision Tree first attempt

# In[57]:


# Create a DecisionTreeRegressor object
tree = DecisionTreeRegressor(max_depth=3)

# Fit the model to the training data
tree.fit(X_train, y_train)


# In[58]:


tree_pred = tree.predict(X_test)

tree_pred


# ### Evaluation Metrics (Decision Tree)

# In[59]:


evaluation_metrics(tree_pred)


# The R-squared metric is much better for this method

# ### Visualisation

# In[60]:


# plot the tree
viz = dtreeviz.model(
    model=tree,
    X_train=X_train,
    y_train=y_train,
    target_name='avg_salary',
    feature_names=cols)


# In[61]:


viz.rtree_feature_space(features=['state_encoded'])


# In[62]:


# viz.rtree_feature_space3D(features=['state_encoded','urgency_encoded'],
#                                  fontsize=10,
#                                  elev=30, azim=20,
#                                  show={'splits', 'title'},
#                                  colors={'tessellation_alpha': .5})


# ### Improvements
# Let's start from tuning model parameters
# 
# `max_depth`: The maximum depth of the tree. Reducing this can help to prevent overfitting.
# `min_samples_split`: The minimum number of samples required to split an internal node. Increasing this can prevent the model from learning too detailed relations which might only hold in the training data.
# `min_samples_leaf`: The minimum number of samples required to be at a leaf node. Similar to min_samples_split, increasing this can help to prevent overfitting.
# `max_features`: The number of features to consider when looking for the best split. Trying different values might lead to better results.

# We will choose the best set of params using GridSearchCV

# In[63]:


# Define the parameter grid
param_grid = {
    'max_depth': [10, 20, 30, 40],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'max_features': ['auto', 'sqrt']
}

# Create a DecisionTreeRegressor object
tree2 = DecisionTreeRegressor()

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=tree2, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Create a new DecisionTreeRegressor object with the best parameters
best_tree = DecisionTreeRegressor(**best_params)

# Fit the new model to the data
best_tree.fit(X_train, y_train)

# Make predictions with the new model
best_pred = best_tree.predict(X_test)


# ### Evaluation Metrics with best params

# In[64]:


evaluation_metrics(best_pred)


# ### Visualisation with best params

# In[65]:


# plot the tree
viz = dtreeviz.model(
    model=best_tree,
    X_train=X_train,
    y_train=y_train,
    target_name='avg_salary',
    feature_names=cols)
viz.rtree_feature_space(features=['state_encoded'])


# In[66]:


residuals_plot(best_pred)


# # Random Forest Regression (ensemble learning method)

# ### Random Forest Application

# In[67]:


randomForest = RandomForestRegressor(n_estimators=100, random_state=123)

randomForest.fit(X_train, y_train)


# In[68]:


rf_pred = randomForest.predict(X_test)


# ### Evaluation Metrics (Random Forest)

# In[69]:


evaluation_metrics(rf_pred)


# # Gradient Boosting Regression

# ### Gradient Boosting Regression Application

# In[70]:


gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=123, loss='squared_error')

gbr.fit(X_train, y_train)


# In[71]:


gbr_pred = gbr.predict(X_test)


# ### Evaluation Metrics (Gradient Boosting Regression)

# In[72]:


evaluation_metrics(gbr_pred)


# # Support Vector Regression

# ### Support Vector Regression Application

# In[73]:


svr = SVR(kernel='linear', C=1.0, epsilon=0.2)
svr.fit(X_train, y_train)


# In[74]:


svr_pred = svr.predict(X_test)


# ### Evaluation Metrics (Support Vector Regression)

# In[75]:


evaluation_metrics(svr_pred)

