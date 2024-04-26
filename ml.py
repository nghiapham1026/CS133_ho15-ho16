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

# Run LinearSVC separately as it would crash the environment if run together with others
svm_model = LinearSVC()
svm_scores = cross_val_score(svm_model, train_prepared_final, strat_train_set['income'], cv=10, n_jobs=-1)
print('SVM Scores:', svm_scores)
reduced_cv_scores['SVM'] = svm_scores

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = [ # From the lecture
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

# Initialize GridSearchCV with the RandomForest classifier
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=10,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all available cores
    verbose=1
)

# Grid Search using 30% of training data since the original dataset takes too long
_, sample_data, _, sample_target = train_test_split(
    train_prepared_final, strat_train_set['income'], test_size=0.3, random_state=42, stratify=strat_train_set['income'])

# Fit GridSearchCV to the sampled training data
grid_search.fit(sample_data, sample_target)

# Extract the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best parameters:", best_params)
print("Best model:", best_model)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load best model from grid search
best_model = grid_search.best_estimator_

# Predict the target variable for the test set
test_predictions = best_model.predict(test_prepared_final)

# Calculate the accuracy of the model on the test set
test_accuracy = accuracy_score(test_y, test_predictions)

# Generate a classification report
test_classification_report = classification_report(test_y, test_predictions)

# Generate a confusion matrix
test_confusion_matrix = confusion_matrix(test_y, test_predictions)

# Print the test accuracy, classification report, and confusion matrix
print("Test Accuracy:", test_accuracy)
print("Classification Score:", test_classification_report)
print("Confusion Matrix:", test_confusion_matrix)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    train_prepared_final, strat_train_set['income'], test_size=0.3, random_state=42, stratify=strat_train_set['income'])

# Hashmap to hold model ROC data
roc_data = {}

# Train each model and calculate ROC curve and AUC
for name, model in reduced_models.items():
    # Train model
    model.fit(X_train, y_train)
    # Predict probabilities for the positive class
    y_scores = model.predict_proba(X_test)[:, 1]
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    roc_data[name] = (fpr, tpr, roc_auc)

# Plotting all ROC curves
plt.figure(figsize=(10, 8))
for name, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, confusion_matrix
import matplotlib.pyplot as plt

# Dictionary to store model predictions
model_predictions = {}

# Fit models and store predictions
for name, model in reduced_models.items():
    model.fit(X_train, y_train)
    # Predict class probabilities for models that support it
    if hasattr(model, "predict_proba"):
        proba_predictions = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
        model_predictions[name] = proba_predictions
    else:
        # Use decision function if predict_proba is not available
        decision_scores = model.decision_function(X_test)
        model_predictions[name] = decision_scores