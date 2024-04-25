import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# Read in data
adults = 'https://raw.githubusercontent.com/csbfx/advpy122-data/master/adult.csv'
data = pd.read_csv(adults, na_values=['?'])

print(data)

# Basic data information
data_info = data.info()

# Display the first few rows
first_few_rows = data.head()

# Print categorical fields and perform value counts
categorical_columns = data.select_dtypes(include=['object']).columns
value_counts = {column: data[column].value_counts() for column in categorical_columns}

print(data_info, first_few_rows, categorical_columns, value_counts)