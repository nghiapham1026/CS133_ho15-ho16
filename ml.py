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