# -- coding: utf-8 --
"""
Created on Wed Mar 19 20:20:45 2025

@author: KSI_Group_5_section_1COMP247Project 
"""

#these are the imports we will use for the model
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Import for Graphs and visualizations
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. Data exploration: a complete review and analysis of the dataset including:
# -----------------------------------------------------------------------------
#data exploring

# Load the dataset from the public URL
url = "https://raw.githubusercontent.com/adornshaju/ksi/refs/heads/main/TOTAL_KSI_6386614326836635957.csv"
Group_data = pd.read_csv(url)
# --------------------------------------------------------------
print(f"Original dataset now has {Group_data.shape[0]} rows and {Group_data.shape[1]} columns.")

# Check for duplicate rows
duplicate_rows = Group_data[Group_data.duplicated(keep=False)]  # Keeps all duplicates

# Count how many times each duplicate row appears (Fixed)
duplicate_counts = duplicate_rows.groupby(duplicate_rows.columns.tolist()).size()
num_duplicates = duplicate_rows.shape[0]

#.shape[0] → Counts the rows (number of duplicates in this case).
if num_duplicates > 0:
    print(f"Found {num_duplicates} duplicate rows.\n")
    print("Duplicate Rows and Their Counts:")
    print(duplicate_counts)  # Shows duplicate rows and their occurrences
else:
    print("No duplicate rows found. The dataset does not contain exact copies.")

# Check for duplicate columns (by data content)
duplicate_columns = Group_data.T.duplicated().sum()
print(f"Found {duplicate_columns} duplicate columns based on identical data.")

# Remove duplicate rows from the dataset
Group_data = Group_data.drop_duplicates()

# Remove duplicate columns from the dataset
Group_data = Group_data.loc[:, ~Group_data.T.duplicated()]

# Confirm removal
print(f"Cleaned dataset now has {Group_data.shape[0]} rows and {Group_data.shape[1]} columns.")
print("Duplicate rows and columns successfully removed.")
# --------------------------------------------------------------
# Display the information about the Dataset
print("\nDataset Information:")
print(Group_data.info())

# Display the first five rows of the dataset
print("\nFirst Five Rows:")
print(Group_data.head())

# Display the names of the columns and types of data
print("\nColumn names and data types: ")
print(Group_data.dtypes)

# Describe the dataset (summary statistics) for numeric columns only
print("\nSummary Statistics: ")
print(Group_data.describe())

# Display the last five rows of the dataset
print("\nUnique Values per column: ")
# Display unique values for each column if the number of unique values is less than or equal to 15
#Identifying categorical-like variables:
# Columns with 15 or fewer unique values are often considered categorical or ordinal in nature (e.g., colors, grades, or rating levels).
for col in Group_data.columns:
   unique_values = Group_data[col].nunique()
   if unique_values <= 15:
         print(col + " : " + str(unique_values) + " - " + str(Group_data[col].unique()))

# Display the last five rows of the dataset
print("\nRanges of Numeric Columns: ")
# Display the range of numeric columns min and max values
for col in Group_data.select_dtypes(include=np.number).columns:
    print(col + " : " + str(Group_data[col].min()) + " - " + str(Group_data[col].max()))

# select the numeric_columns from the dataset
#Here we are filtering all the numeric columns from the dataset before the calculations
numeric_cols = Group_data.select_dtypes(include=['int64', 'float64'])

#displays the mean
print("\nMean (Average) of Numeric Columns:")
print(numeric_cols.mean())

# Calculate the median for numeric columns
print("\nMedian of Numeric Columns:")
print(numeric_cols.median())

# Calculate the standard deviation for numeric columns
print("\nStandard Deviation of Numeric Columns:")
print(numeric_cols.std())

# Calculate correlations between numeric columns
# print("\n🔗 Correlation Between Numeric Columns:")
# Group_data.corr()

# Calculate correlations between numeric columns
correlations = numeric_cols.corr()
#we use the correlation to see how the variables are related to each other and how they can be used to predict the target variable in the dataset.
#The correlation coefficient ranges from -1 to 1. If the value is close to 1, it means that there is a strong positive correlation between the two variables.

print("\nCorrelation Between Numeric Columns:")
print(correlations)

# Calculate correlations between numeric columns using the Spearman method
spearman_corr = numeric_cols.corr(method='spearman')
print("\nSpearman Correlation Between Numeric Columns:")
print(spearman_corr)

#differences between the normal correlation:
# and spearman correlation Measures the linear relationship between two variables.
# Spearman  Measures the monotonic relationship between two variables (whether the relationship is consistently increasing or decreasing,
# but not necessarily linear).

#display the missing values in the dataset
print("\nNumber of missing values: ")
missing_values = Group_data.isnull().sum()

#calculate the percentage of missing values, this is the formula
missing_percentage = (missing_values / len(Group_data)) * 100

#displaying the results and storing them in a dataframe for better visualization
missing_data_summary = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})

#display the missing values if there are any in the dataset and sort them in descending order to see the most missing values
print("\n Missing Data Summary:")
print(missing_data_summary[missing_data_summary['Missing Values'] > 0].sort_values('Percentage', ascending=False))

# -----------------------------------------------------------------------------
# Graphs and visualizations

# Set seaborn style
sns.set(style="whitegrid")

# Plot the distribution of the target variable (if found)
target = Group_data["ACCLASS"].apply(lambda x: 1 if x == "Fatal" else 0)

# Plot the cleaned distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=target.map({0: "Non-Fatal", 1: "Fatal"}))
plt.title("Count of Fatal vs Non-Fatal Collisions")
plt.xlabel("Collision Severity")
plt.ylabel("Number of Cases")
plt.tight_layout()
plt.show()

# Count values
counts = target.value_counts().sort_index()  # 0: Non-Fatal, 1: Fatal
labels = ["Non-Fatal", "Fatal"]
# Pie chart
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=["skyblue", "salmon"])
plt.title("Fatal vs Non-Fatal Collisions (Pie Chart)")
plt.axis('equal')
plt.show()

# Count values using target
target_counts = target.value_counts().sort_index()  # 0: Non-Fatal, 1: Fatal
# Bar plot using pandas and target variable
target_counts.plot(kind='bar', color=["skyblue", "salmon"])
plt.xticks(ticks=[0, 1], labels=["Non-Fatal", "Fatal"], rotation=0)
plt.title("Fatal vs Non-Fatal Collisions")
plt.ylabel("Number of Collisions")
plt.xlabel("Collision Type")
plt.tight_layout()
plt.show()

# Get numeric columns from Group_data
numeric_cols = Group_data.select_dtypes(include=['int64', 'float64'])
# Create a temporary DataFrame that includes the target
temp_df = numeric_cols.copy()
temp_df['Target'] = target  # Add target column temporarily
# Generate the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(temp_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap (Including Fatal/Non-Fatal Target)")
plt.show()

# No need to use 'target' variable
# Missing values heatmap 
plt.figure(figsize=(10, 6))
sns.heatmap(Group_data.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()

# --------------------------------------

plt.figure(figsize=(12, 5))
sns.countplot(x="LIGHT", data=Group_data, hue=target, order=Group_data["LIGHT"].value_counts().index)
plt.xticks(rotation=60, ha='right', fontsize=6)
plt.title("Accidents by Light Conditions (Fatal vs Non-Fatal)")
plt.xlabel("Light Condition", fontsize=8)
plt.ylabel("Number of Accidents", fontsize=8)
plt.legend(title="Collision Type")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x="RDSFCOND", data=Group_data, hue=target, order=Group_data["RDSFCOND"].value_counts().index)
plt.xticks(rotation=60, ha='right', fontsize=6)
plt.title("Accidents by Road Surface Conditions (Fatal vs Non-Fatal)")
plt.xlabel("Road Surface Condition", fontsize=8)
plt.ylabel("Number of Accidents", fontsize=8)
plt.legend(title="Collision Type")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
sns.countplot(x="DRIVACT", data=Group_data, hue=target)
plt.title("Accidents by Driver Action (Fatal vs Non-Fatal)")
plt.xticks(rotation=60, ha='right', fontsize=6)
plt.xlabel("Driver Action", fontsize=8)
plt.ylabel("Number of Accidents", fontsize=8)
plt.legend(title="Collision Type")
plt.tight_layout()
plt.show()


pd.crosstab(Group_data["DRIVCOND"], target, normalize='index').plot(kind='bar', stacked=True)
plt.title("Fatal vs. Non-Fatal Collisions by Driver Condition")
plt.ylabel("Proportion")
plt.xlabel("Driver Condition")
plt.xticks(rotation=60, ha='right', fontsize=6)
plt.legend(title="Collision Type", labels=["Non-Fatal", "Fatal"])
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
sns.countplot(x="PEDACT", data=Group_data, hue=target, order=Group_data["PEDACT"].value_counts().index)
plt.xticks(rotation=60, ha='right', fontsize=6)
plt.title("Accidents by Pedestrian Action (Fatal vs Non-Fatal)")
plt.xlabel("Pedestrian Action", fontsize=8)
plt.ylabel("Number of Accidents", fontsize=8)
plt.legend(title="Collision Type", labels=["Non-Fatal", "Fatal"])
plt.tight_layout()
plt.show()

pd.crosstab(Group_data["PEDACT"], target, normalize='index').plot(kind='bar', stacked=True)
plt.title("Fatal vs. Non-Fatal Collisions by Pedestrian Action")
plt.ylabel("Proportion", fontsize=8)
plt.xlabel("Pedestrian Action", fontsize=8)
plt.xticks(rotation=60, ha='right', fontsize=6)
plt.legend(title="Collision Type", labels=["Non-Fatal", "Fatal"])
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 2. DATA MODELLING
# -----------------------------------------------------------------------------

# 2.1. Data transformations – Handling missing data, categorical data, normalization, standardization
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd
import numpy as np

# Step 1: Define Target
y = Group_data["ACCLASS"].apply(lambda x: 1 if x == "Fatal" else 0)

# Step 2: Identify columns
categorical_cols = Group_data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = Group_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove target from categorical columns
if "ACCLASS" in categorical_cols:
    categorical_cols.remove("ACCLASS")

# Step 3: Separate inputs (X)
X = Group_data[categorical_cols + numerical_cols]

# Step 4: Define pipelines
# Numerical
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Step 5: Create preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# Step 6: Transform the data
X_preprocessed = preprocessor.fit_transform(X)

# Step 7: Get feature names
cat_features = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_cols)
feature_names_combined = np.concatenate([numerical_cols, cat_features])

# Step 8: Split preprocessed data into X_num and X_cat
X_num = X_preprocessed[:, :len(numerical_cols)]
X_cat = X_preprocessed[:, len(numerical_cols):]

# Step 9: Select top 10 categorical features only
selector = SelectKBest(score_func=f_classif, k=10)
X_cat_selected = selector.fit_transform(X_cat, y)
selected_cat_features = cat_features[selector.get_support()]

# Step 10: Combine numerical and selected categorical features
X_selected = np.concatenate([X_num, X_cat_selected], axis=1)
final_feature_names = numerical_cols + selected_cat_features.tolist()

# Step 11: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=25
)

# Step 12: Convert to DataFrame
X_train = pd.DataFrame(X_train, columns=final_feature_names)
X_test = pd.DataFrame(X_test, columns=final_feature_names)

# Step 13: Apply SMOTE
print("\nClass distribution before SMOTE:", Counter(y_train))
smote = SMOTE(random_state=25)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:", Counter(y_train_balanced))

# Debugging info
print("\nColumns in X_train_balanced after SMOTE:", X_train_balanced.columns.tolist())
print("Columns in X_test:", X_test.columns.tolist())

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2.2. Feature selection – (Choosing relevant NUMERICAL and CATEGORICAL features)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

# Step 1: Define the target variable
y = Group_data["ACCLASS"].apply(lambda x: 1 if x == "Fatal" else 0)

# Step 2: Identify categorical and numerical columns
categorical_cols = Group_data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = Group_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Step 3: Remove target from X
if "ACCLASS" in categorical_cols:
    categorical_cols.remove("ACCLASS")
X = Group_data[categorical_cols + numerical_cols]

# Step 4: Create pipelines
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # sparse_output for sklearn >= 1.2
])
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Step 5: Create full preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_pipeline, categorical_cols),
    ("num", numerical_pipeline, numerical_cols)
])

# Step 6: Fit + transform X
X_preprocessed = preprocessor.fit_transform(X)

# Step 7: Get all transformed feature names
encoded_cat_features = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
all_feature_names = np.concatenate([encoded_cat_features, numerical_cols])

# Step 8: Select only categorical features for SelectKBest
X_cat_transformed = X_preprocessed[:, :len(encoded_cat_features)]
X_num_transformed = X_preprocessed[:, len(encoded_cat_features):]

# Step 9: Select top 10 categorical features only
selector = SelectKBest(score_func=f_classif, k=min(10, X_cat_transformed.shape[1]))
X_cat_selected = selector.fit_transform(X_cat_transformed, y)

# Step 10: Final selected features = top 10 categorical + all numerical
selected_cat_features = encoded_cat_features[selector.get_support()]
final_feature_names = list(selected_cat_features) + numerical_cols
X_selected = np.concatenate([X_cat_selected, X_num_transformed], axis=1)

# Step 11: Print feature selection results
print("\nSelected Top K Categorical Features Used for Training:")
for col in selected_cat_features:
    print("-", col)

selected_scores = selector.scores_[selector.get_support()]
selected_feature_df = pd.DataFrame({
    "Feature": selected_cat_features,
    "F-Score": selected_scores
}).sort_values(by="F-Score", ascending=False)

print("\nF-Scores of Selected Categorical Features:")
print(selected_feature_df)

# Optional: Print the least useful features (full list sorted ascending)
full_feature_scores = pd.DataFrame({
    "Feature": encoded_cat_features,
    "F-Score": selector.scores_
}).sort_values(by="F-Score", ascending=True)

print("\nBottom Categorical Features by F-Score (Least Useful):")
print(full_feature_scores.head(10))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2.3. Train, Test data splitting – Using train_test_split
from sklearn.model_selection import train_test_split

# Splitting data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=25, stratify=y)

print(f"Data Split: Training Set = {X_train.shape[0]} rows, Testing Set = {X_test.shape[0]} rows.")

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2.4. Managing imbalanced classes – Oversampling / Undersampling
from imblearn.over_sampling import SMOTE
from collections import Counter

# Convert X_train back to a DataFrame if it's a NumPy array (before SMOTE)
if isinstance(X_train, np.ndarray):  
    print("Warning: X_train is a NumPy array. Converting back to DataFrame...")
    X_train = pd.DataFrame(X_train, columns=final_feature_names)

# Save feature names before SMOTE
feature_names = X_train.columns

# Check class distribution before balancing (Training Data)
print("\nClass distribution in training data before balancing:", Counter(y_train))

# Apply SMOTE for oversampling (Training Data)
smote = SMOTE(random_state=25)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Convert X_train_balanced back to DataFrame with original column names
X_train_balanced = pd.DataFrame(X_train_balanced, columns=feature_names)

# Check class distribution after balancing (Training Data)
print("\nClass distribution in training data after SMOTE balancing:", Counter(y_train_balanced))

# Apply SMOTE to Testing Data (not recommended)
print("\nWARNING: Applying SMOTE to testing data may distort evaluation metrics.")
print("This is not a standard practice and may lead to biased results.")

# Convert X_test to DataFrame if needed
if isinstance(X_test, np.ndarray):
    X_test = pd.DataFrame(X_test, columns=final_feature_names)

# Check class distribution in real test set
print("\nClass distribution in testing data (no SMOTE applied):", Counter(y_test))

# Verify columns before transforming
print("\nColumns in X_train_balanced after SMOTE and feature selection:", X_train_balanced.columns.tolist())
print("Columns in X_test after feature selection (real test set):", X_test.columns.tolist())
# -----------------------------------------------------------------------------

# 2.5. Using Pipelines to streamline preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# Ensure input is in DataFrame format
if isinstance(X_train_balanced, np.ndarray):
    X_train_balanced = pd.DataFrame(X_train_balanced, columns=feature_names)

if isinstance(X_test, np.ndarray):
    X_test = pd.DataFrame(X_test, columns=feature_names)

# Separate numerical and categorical columns from balanced training data
numerical_cols = X_train_balanced.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X_train_balanced.select_dtypes(include=["object"]).columns.tolist()

print("\nNumerical Columns for Pipeline:", numerical_cols)
print("Categorical Columns for Pipeline:", categorical_cols)

# Preprocessing pipeline for numerical features
numerical_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Preprocessing pipeline for categorical features
categorical_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_preprocessor, numerical_cols),
        ("cat", categorical_preprocessor, categorical_cols)
    ]
)

# Full preprocessing pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor)
])

# Fit and transform training data
X_train_transformed = pipeline.fit_transform(X_train_balanced)

# Transform test data
X_test_transformed = pipeline.transform(X_test)

# Output results
print("\nPreprocessing pipeline applied successfully.")
print("Shape of Transformed Training Data:", X_train_transformed.shape)
print("Shape of Transformed Test Data:", X_test_transformed.shape)

print("\nPreview of Transformed Training Data:")
print(pd.DataFrame(X_train_transformed).head())

print("\nPreview of Transformed Test Data:")
print(pd.DataFrame(X_test_transformed).head())

# Optional: Get feature names after transformation (if supported by your sklearn version)
try:
    feature_names_out = pipeline.named_steps["preprocessing"].get_feature_names_out()
    print("\nTransformed Feature Names:")
    print(feature_names_out)
except:
    print("\nNote: `get_feature_names_out()` requires scikit-learn 1.2 or higher.")

print("\nConfirming input shape matches feature names:")
print("Transformed X_train shape:", X_train_transformed.shape)
print("Feature names length:", len(feature_names_out))

# -----------------------------------------------------------------------------
# 3. Predictive model building with Transformed Features
# -----------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import joblib
import pandas as pd

model_scores = {}

plt.figure(figsize=(8, 6))  # ROC plot setup

# 1. Logistic Regression
# -------------------------------------------------------------
# LOGISTIC REGRESSION - BEFORE TUNING (Baseline)
# -------------------------------------------------------------
baseline_log_model = LogisticRegression(random_state=25)  # Default settings
baseline_log_model.fit(X_train_balanced, y_train_balanced)
baseline_log_pred = baseline_log_model.predict(X_test)
baseline_log_proba = baseline_log_model.predict_proba(X_test)[:, 1]

print("\n---------- Logistic Regression (Before Tuning) ----------")
print("Confusion Matrix:\n", confusion_matrix(y_test, baseline_log_pred))
print("Classification Report:\n", classification_report(y_test, baseline_log_pred))
fpr, tpr, _ = roc_curve(y_test, baseline_log_proba)
plt.plot(fpr, tpr, label="LogReg (Before) AUC = {:.2f}".format(auc(fpr, tpr)))

# Save baseline model
joblib.dump({
    "model": baseline_log_model,
    "features": final_feature_names
}, "logistic_regression_baseline.pkl")
print("Saved as 'logistic_regression_baseline.pkl'")

# -------------------------------------------------------------
# LOGISTIC REGRESSION - AFTER TUNING
# -------------------------------------------------------------
tuned_log_model = LogisticRegression(
    penalty='l2',               # L2 regularization
    solver='liblinear',         # More stable on small/mid datasets
    class_weight='balanced',    # Handle imbalanced classes
    max_iter=1000,
    random_state=25
)
tuned_log_model.fit(X_train_balanced, y_train_balanced)
tuned_log_pred = tuned_log_model.predict(X_test)
tuned_log_proba = tuned_log_model.predict_proba(X_test)[:, 1]

model_scores["Logistic Regression"] = {
    "Accuracy": accuracy_score(y_test, tuned_log_pred),
    "Precision": precision_score(y_test, tuned_log_pred),
    "Recall": recall_score(y_test, tuned_log_pred),
    "F1 Score": f1_score(y_test, tuned_log_pred)
}

print("\n---------- Logistic Regression (After Tuning) ----------")
print("Confusion Matrix:\n", confusion_matrix(y_test, tuned_log_pred))
print("Classification Report:\n", classification_report(y_test, tuned_log_pred))
fpr, tpr, _ = roc_curve(y_test, tuned_log_proba)
plt.plot(fpr, tpr, label="LogReg (Tuned) AUC = {:.2f}".format(auc(fpr, tpr)))

# Save tuned model
joblib.dump({
    "model": tuned_log_model,
    "features": final_feature_names
}, "logistic_regression_model.pkl")
print("Logistic Regression model saved as 'logistic_regression_model.pkl'")

# 2. Random Forest
# -------------------------------------------------------------
# RANDOM FOREST - BEFORE TUNING (Baseline)
# -------------------------------------------------------------
baseline_rf_model = RandomForestClassifier(random_state=25)  # Default hyperparameters
baseline_rf_model.fit(X_train_balanced, y_train_balanced)
baseline_rf_pred = baseline_rf_model.predict(X_test)
baseline_rf_proba = baseline_rf_model.predict_proba(X_test)[:, 1]

print("\n---------- Random Forest (Before Tuning) ----------")
print("Confusion Matrix:\n", confusion_matrix(y_test, baseline_rf_pred))
print("Classification Report:\n", classification_report(y_test, baseline_rf_pred))
fpr, tpr, _ = roc_curve(y_test, baseline_rf_proba)
plt.plot(fpr, tpr, label="RF (Before) AUC = {:.2f}".format(auc(fpr, tpr)))

# Save baseline model
joblib.dump({
    "model": baseline_rf_model,
    "features": final_feature_names
}, "random_forest_baseline.pkl")
print("Saved as 'random_forest_baseline.pkl'")
print("----------------------------------------")

# -------------------------------------------------------------
# RANDOM FOREST - AFTER TUNING (Improved Hyperparameters)
# -------------------------------------------------------------
tuned_rf_model = RandomForestClassifier(
    n_estimators=300,        # Increased number of trees
    max_depth=25,            # Limit tree depth to prevent overfitting
    class_weight='balanced', # Address class imbalance
    random_state=25
)
tuned_rf_model.fit(X_train_balanced, y_train_balanced)
tuned_rf_pred = tuned_rf_model.predict(X_test)
tuned_rf_proba = tuned_rf_model.predict_proba(X_test)[:, 1]

model_scores["Random Forest"] = {
    "Accuracy": accuracy_score(y_test, tuned_rf_pred),
    "Precision": precision_score(y_test, tuned_rf_pred),
    "Recall": recall_score(y_test, tuned_rf_pred),
    "F1 Score": f1_score(y_test, tuned_rf_pred)
}

print("\n---------- Random Forest (After Tuning) ----------")
print("Confusion Matrix:\n", confusion_matrix(y_test, tuned_rf_pred))
print("Classification Report:\n", classification_report(y_test, tuned_rf_pred))
fpr, tpr, _ = roc_curve(y_test, tuned_rf_proba)
plt.plot(fpr, tpr, label="RF (Tuned) AUC = {:.2f}".format(auc(fpr, tpr)))

# Save tuned model
joblib.dump({
    "model": tuned_rf_model,
    "features": final_feature_names
}, "random_forest_model.pkl")
print("Random Forest model saved as 'random_forest_model.pkl'")
print("----------------------------------------")

# 3. SVM
# -------------------------------------------------------------
# SVM - BEFORE TUNING (Baseline)
# -------------------------------------------------------------
baseline_svm_model = SVC(probability=True, random_state=25)
baseline_svm_model.fit(X_train_balanced, y_train_balanced)
baseline_svm_pred = baseline_svm_model.predict(X_test)
baseline_svm_proba = baseline_svm_model.predict_proba(X_test)[:, 1]

print("\n---------- SVM (Before Tuning) ----------")
print("Confusion Matrix:\n", confusion_matrix(y_test, baseline_svm_pred))
print("Classification Report:\n", classification_report(y_test, baseline_svm_pred))
fpr, tpr, _ = roc_curve(y_test, baseline_svm_proba)
plt.plot(fpr, tpr, label="SVM (Before) AUC = {:.2f}".format(auc(fpr, tpr)))

joblib.dump({
    "model": baseline_svm_model,
    "features": final_feature_names
}, "svm_baseline.pkl")
print("Saved as 'svm_baseline.pkl'")

# -------------------------------------------------------------
# SVM - AFTER TUNING
# -------------------------------------------------------------
tuned_svm_model = SVC(
    kernel='rbf',               # RBF = non-linear kernel
    gamma='scale',              # Uses 1 / (n_features * X.var()) as gamma
    class_weight='balanced',    # Fix imbalance in classes
    probability=True,           # Enables ROC + predict_proba
    C=1.0,                      # Soft margin cost (optional)
    random_state=25
)

tuned_svm_model.fit(X_train_balanced, y_train_balanced)
tuned_svm_pred = tuned_svm_model.predict(X_test)
tuned_svm_proba = tuned_svm_model.predict_proba(X_test)[:, 1]

model_scores["SVM"] = {
    "Accuracy": accuracy_score(y_test, tuned_svm_pred),
    "Precision": precision_score(y_test, tuned_svm_pred),
    "Recall": recall_score(y_test, tuned_svm_pred),
    "F1 Score": f1_score(y_test, tuned_svm_pred)
}

print("\n---------- SVM (After Tuning) ----------")
print("Confusion Matrix:\n", confusion_matrix(y_test, tuned_svm_pred))
print("Classification Report:\n", classification_report(y_test, tuned_svm_pred))
fpr, tpr, _ = roc_curve(y_test, tuned_svm_proba)
plt.plot(fpr, tpr, label="SVM (Tuned) AUC = {:.2f}".format(auc(fpr, tpr)))

joblib.dump({
    "model": tuned_svm_model,
    "features": final_feature_names
}, "svm_model.pkl")
print("SVM model saved as 'svm_model.pkl'")
print("----------------------------------------")

# 4. Neural Network
# -----------------------------------------------------------------------------
# Neural Network - BEFORE TUNING (baseline)
# -----------------------------------------------------------------------------
baseline_nn_model = MLPClassifier(max_iter=1000, random_state=25)
baseline_nn_model.fit(X_train_balanced, y_train_balanced)
baseline_nn_pred = baseline_nn_model.predict(X_test)
baseline_nn_proba = baseline_nn_model.predict_proba(X_test)[:, 1]

print("\n---------- Neural Network (Before Tuning) ----------")
print("Confusion Matrix:\n", confusion_matrix(y_test, baseline_nn_pred))
print("Classification Report:\n", classification_report(y_test, baseline_nn_pred))
fpr, tpr, _ = roc_curve(y_test, baseline_nn_proba)
plt.plot(fpr, tpr, label="Neural Net (Before) AUC = {:.2f}".format(auc(fpr, tpr)))

joblib.dump({
    "model": baseline_nn_model,
    "features": final_feature_names
}, "neural_network_baseline.pkl")
print("Saved as 'neural_network_baseline.pkl'")


# -----------------------------------------------------------------------------
# Neural Network - AFTER TUNING
# -----------------------------------------------------------------------------
nn_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Two hidden layers: 100 to 50 neurons
    alpha=0.0005,                  # Adds more L2 regularization to reduce overfitting from 0.0001
    learning_rate='adaptive',      # Adjusts learning rate based on training performance
    max_iter=1000,                 # Maximum epochs to converge
    random_state=25
)
nn_model.fit(X_train_balanced, y_train_balanced)
nn_pred = nn_model.predict(X_test)
nn_proba = nn_model.predict_proba(X_test)[:, 1]

model_scores["Neural Network"] = {
    "Accuracy": accuracy_score(y_test, nn_pred),
    "Precision": precision_score(y_test, nn_pred),
    "Recall": recall_score(y_test, nn_pred),
    "F1 Score": f1_score(y_test, nn_pred)
}

print("\n---------- Neural Network (After Tuning) ----------")
print("Confusion Matrix:\n", confusion_matrix(y_test, nn_pred))
print("Classification Report:\n", classification_report(y_test, nn_pred))
fpr, tpr, _ = roc_curve(y_test, nn_proba)
plt.plot(fpr, tpr, label="Neural Net (Tuned) AUC = {:.2f}".format(auc(fpr, tpr)))

joblib.dump({
    "model": nn_model,
    "features": final_feature_names
}, "neural_network_model.pkl")
print("Neural Network model saved as 'neural_network_model.pkl'")
print("----------------------------------------")

# 5. KNN
# -----------------------------------------------------------------------------
# KNN - BEFORE TUNING (baseline)
# -----------------------------------------------------------------------------
baseline_knn = KNeighborsClassifier(n_neighbors=5)
baseline_knn.fit(X_train_balanced, y_train_balanced)
baseline_knn_pred = baseline_knn.predict(X_test)
baseline_knn_proba = baseline_knn.predict_proba(X_test)[:, 1]

print("\n---------- KNN (Before Tuning) ----------")
print("Confusion Matrix:\n", confusion_matrix(y_test, baseline_knn_pred))
print("Classification Report:\n", classification_report(y_test, baseline_knn_pred))
fpr, tpr, _ = roc_curve(y_test, baseline_knn_proba)
plt.plot(fpr, tpr, label="KNN (Before) AUC = {:.2f}".format(auc(fpr, tpr)))

joblib.dump({
    "model": baseline_knn,
    "features": final_feature_names
}, "knn_baseline_model.pkl")
print("Saved as 'knn_baseline_model.pkl'")


# -----------------------------------------------------------------------------
# KNN - AFTER TUNING
# -----------------------------------------------------------------------------
knn_model = KNeighborsClassifier(
    n_neighbors=7,        # Try different neighbor counts / Number of neighbors to vote. More neighbors = smoother boundary.
    weights='distance',   # Weight closer neighbors more / Closer neighbors have more influence (helps accuracy).
    p=2                   # L2 norm (Euclidean). If p=1, it uses Manhattan distance.
)
knn_model.fit(X_train_balanced, y_train_balanced)
knn_pred = knn_model.predict(X_test)
knn_proba = knn_model.predict_proba(X_test)[:, 1]

model_scores["KNN"] = {
    "Accuracy": accuracy_score(y_test, knn_pred),
    "Precision": precision_score(y_test, knn_pred),
    "Recall": recall_score(y_test, knn_pred),
    "F1 Score": f1_score(y_test, knn_pred)
}

print("\n---------- KNN (After Tuning) ----------")
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print("Classification Report:\n", classification_report(y_test, knn_pred))
fpr, tpr, _ = roc_curve(y_test, knn_proba)
plt.plot(fpr, tpr, label="KNN (Tuned) AUC = {:.2f}".format(auc(fpr, tpr)))

joblib.dump({
    "model": knn_model,
    "features": final_feature_names
}, "knn_model.pkl")
print("KNN model saved as 'knn_model.pkl'")
print("----------------------------------------")

# -----------------------------------------------------------------------------
# 4.1 Final ROC Curve
# -----------------------------------------------------------------------------
plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 4.2 Final Model Summary
# -----------------------------------------------------------------------------
results_df = pd.DataFrame(model_scores).T.sort_values("F1 Score", ascending=False)

print("\n")
print("-------------------------------------------------------------")
print("Model Performance Summary (sorted by F1 Score):")
print(results_df)
print("-------------------------------------------------------------")

best_model = results_df.index[0]

print("\n")
print("-------------------------------------------------------------")
print(f"Recommended Model: {best_model} (Based on highest F1 Score)")
print("-------------------------------------------------------------")


# print("Top 5 STREET1 values used the most:")
# print(Group_data["STREET1"].value_counts().head(5))

# print("Top 5 STREET2 values used the most:")
# print(Group_data["STREET2"].value_counts().head(5))

# print("Top 5 OFFSET values used the most:")
# print(Group_data["OFFSET"].value_counts().head(5))