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