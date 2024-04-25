import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# Read in data
adults = 'https://raw.githubusercontent.com/csbfx/advpy122-data/master/adult.csv'
data = pd.read_csv(adults, na_values=['?'])

print(data)