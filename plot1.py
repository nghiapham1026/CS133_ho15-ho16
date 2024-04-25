import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

adults = 'https://raw.githubusercontent.com/csbfx/advpy122-data/master/adult.csv'
data = pd.read_csv(adults, na_values=['?'])

# Setting up the figure size and subplots
plt.figure(figsize=(15, 10))

# Histograms for numerical fields
num_columns = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
for i, column in enumerate(num_columns, 1):
    plt.subplot(3, 2, i)
    sns.histplot(data[column], kde=False, bins=30)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()