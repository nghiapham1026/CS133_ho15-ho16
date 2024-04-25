import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

adults = 'https://raw.githubusercontent.com/csbfx/advpy122-data/master/adult.csv'
data = pd.read_csv(adults, na_values=['?'])

categorical_columns = data.select_dtypes(include=['object']).columns

# Identify categorical fields and perform value counts
categorical_counts = {column: data[column].value_counts() for column in categorical_columns if column in data}

# Show bar charts for a subset of categorical columns due to space limit
plt.figure(figsize=(15, 12))
selected_categories = ['workclass', 'education', 'marital-status', 'occupation']
for i, column in enumerate(selected_categories, 1):
    plt.subplot(2, 2, i)
    sns.barplot(x=categorical_counts[column].index, y=categorical_counts[column].values)
    plt.title(f'Counts of {column}')
    plt.xticks(rotation=90)
    plt.xlabel(column)
    plt.ylabel('Counts')

plt.tight_layout()
plt.show()