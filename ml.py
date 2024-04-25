from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

adults = 'https://raw.githubusercontent.com/csbfx/advpy122-data/master/adult.csv'
data = pd.read_csv(adults, na_values=['?'])

# Replace '?' with NaN
data.replace('?', pd.NA, inplace=True)

# Map labels to binary format
data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})

# Prepare stratified sampling
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['income']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

# Separating predictors and the target variable
train_x = strat_train_set.drop('income', axis=1)
train_y = strat_train_set['income']
test_x = strat_test_set.drop('income', axis=1)
test_y = strat_test_set['income']

# Columns for transformations
num_attribs = train_x.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_attribs = train_x.select_dtypes(include=['object']).columns.tolist()

# Pipeline for numerical attributes
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

# Pipeline for categorical attributes
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

# Full pipeline for all transformations
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# Apply transformations
train_x = full_pipeline.fit_transform(train_x)
test_x = full_pipeline.transform(test_x)

# Define a pipeline for processing numerical attributes with median imputation and standard scaling
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

# Define a pipeline for processing categorical attributes with most frequent imputation and one-hot encoding
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

# Combine numerical and categorical pipelines into a single column transformer
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# Apply the full pipeline to transform training and testing data
train_x = full_pipeline.fit_transform(train_x)
test_x = full_pipeline.transform(test_x)

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Adding interaction terms between 'age' and 'hours-per-week'
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # Nothing else to do

    def transform(self, X):
        age_hours_interaction = X[:, num_attribs.index('age')] * X[:, num_attribs.index('hours-per-week')]
        return np.c_[X, age_hours_interaction]

# Updating the numerical pipeline to include the attribute adder
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),  # fill in missing values with median
    ('attribs_adder', CombinedAttributesAdder()),   # adding new feature
    ('std_scaler', StandardScaler()),               # scale features
])

# Full pipeline remains unchanged in structure but will now include the updated num_pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# Apply the full pipeline to the training and testing dataset
train_prepared_final = full_pipeline.fit_transform(strat_train_set)
test_prepared_final = full_pipeline.transform(strat_test_set)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC

# Initialize models
reduced_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    #'SVM': SVC() Too intensive, crashing the environment
}

# Perform 10-fold cross-validation for the reduced set of models
reduced_cv_scores = {}
for name, model in reduced_models.items():
    scores = cross_val_score(model, train_prepared_final, strat_train_set['income'], cv=10, n_jobs=-1)
    reduced_cv_scores[name] = scores