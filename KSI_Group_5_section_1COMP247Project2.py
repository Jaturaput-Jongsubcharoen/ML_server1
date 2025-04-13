# -*- coding: utf-8 -*-
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
file_path = r"D:\group\TOTAL_KSI_6386614326836635957.csv"
Group_data = pd.read_csv(file_path)
# --------------------------------------------------------------
print(f"Original dataset now has {Group_data.shape[0]} rows.")
# Find and count duplicate rows (excluding the first occurrence)
duplicate_rows = Group_data[Group_data.duplicated(keep=False)]  # Keeps all duplicates

# Count how many times each duplicate appears (Fixed)
duplicate_counts = duplicate_rows.groupby(duplicate_rows.columns.tolist()).size()

# Print the result
num_duplicates = duplicate_rows.shape[0]
#.shape[0] → Counts the rows (number of duplicates in this case).

if num_duplicates > 0:
    print(f"Found {num_duplicates} duplicate rows.\n")
    print("Duplicate Rows and Their Counts:")
    print(duplicate_counts)  # Shows duplicate rows and their occurrences
else:
    print("No duplicate rows found. The dataset does not contain exact copies.")
    
# Remove duplicate rows from the dataset
Group_data = Group_data.drop_duplicates()

# Confirm removal
print(f"Cleaned dataset now has {Group_data.shape[0]} rows.")
print("Duplicate rows successfully removed.")
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
sns.countplot(x=target)
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

# Bin TIME into categories (e.g., Morning, Afternoon, Evening, Night)
binned_time = Group_data['TIME_CAT'] = pd.cut(Group_data['TIME'], bins=[0, 600, 1200, 1800, 2400], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
sns.countplot(x=binned_time, hue=target)


# -----------------------------------------------------------------------------
# 2. DATA MODELLING
# -----------------------------------------------------------------------------

# 2.1. Data transformations – Handling missing data, categorical data, normalization, standardization
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Identify categorical and numerical columns
categorical_cols = Group_data.select_dtypes(include=['object']).columns
numerical_cols = Group_data.select_dtypes(include=['int64', 'float64']).columns

# Handling missing numerical data (imputation)
num_imputer = SimpleImputer(strategy="median")

# Handling categorical data (encoding)
cat_imputer = SimpleImputer(strategy="most_frequent")  # Fill missing categorical values
one_hot_encoder = OneHotEncoder(handle_unknown="ignore")

# Normalization (MinMax Scaling) & Standardization (Standard Scaler)
scaler = StandardScaler()  # Change to MinMaxScaler() if needed

# Column Transformer: Applies transformations to different column types
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", num_imputer), ("scaler", scaler)]), numerical_cols),
    ("cat", Pipeline([("imputer", cat_imputer), ("encoder", one_hot_encoder)]), categorical_cols)
])

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2.2. Feature selection – Choosing relevant columns
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd

# Define input features and target variable
# Use "ACCLASS" as the target column (based on earlier parts of the script)
X = Group_data.drop(columns=["ACCLASS"], errors="ignore")  # Drop target column
y = Group_data["ACCLASS"].apply(lambda x: 1 if x == "Fatal" else 0)  # Convert target to binary (1: Fatal, 0: Non-Fatal)

# Identify non-numeric columns
non_numeric_cols = X.select_dtypes(exclude=['int64', 'float64']).columns

# Print columns that need conversion or removal
if len(non_numeric_cols) > 0:
    print("\nNon-numeric columns found (these must be removed or converted):")
    print(non_numeric_cols)

# Drop non-numeric columns (like timestamps) before feature selection
X_clean = X.drop(columns=non_numeric_cols, errors="ignore")

# Check for NaN values before imputation
print("\nChecking for missing values before imputation:")
print(X_clean.isnull().sum()[X_clean.isnull().sum() > 0])

# Drop irrelevant columns (if they exist)
irrelevant_cols = ["OBJECTID", "INDEX", "ACCNUM"]
X_clean = X_clean.drop(columns=irrelevant_cols, errors="ignore")

# Handle missing values by filling NaN with the median (for numerical columns)
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X_clean), columns=X_clean.columns)

# Check if any NaN values remain after imputation
if np.isnan(X_imputed).sum().sum() > 0:
    print("\nERROR: NaN values still present after imputation!")
    print(X_imputed.isnull().sum()[X_imputed.isnull().sum() > 0])
else:
    print("\nAll missing values successfully handled.")

# Ensure all values are finite (no NaN or Inf)
X_imputed = X_imputed.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
X_imputed = X_imputed.dropna()  # Drop any rows that still contain NaN

# Feature selection using ANOVA F-test (Select Top 10 Features)
feature_selector = SelectKBest(score_func=f_classif, k=min(10, X_imputed.shape[1]))  # Ensure k ≤ total features
X_selected = feature_selector.fit_transform(X_imputed, y.loc[X_imputed.index])  # Ensure y is aligned

# Print selected feature scores
feature_scores = pd.DataFrame({"Feature": X_clean.columns, "Score": feature_selector.scores_})
print("\nFeature Selection Scores:")
print(feature_scores.sort_values(by="Score", ascending=False))

# -----------------------------------------------------------------------------
# 2.3. Train, Test data splitting – Using train_test_split
from sklearn.model_selection import train_test_split

# Splitting data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

print(f"Data Split: Training Set = {X_train.shape[0]} rows, Testing Set = {X_test.shape[0]} rows.")

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2.4. Managing imbalanced classes – Oversampling / Undersampling
from imblearn.over_sampling import SMOTE
from collections import Counter

# Convert X_train back to a DataFrame if it's a NumPy array (before SMOTE)
if isinstance(X_train, np.ndarray):  
    print("Warning: X_train is a NumPy array. Converting back to DataFrame...")
    X_train = pd.DataFrame(X_train, columns=X_clean.columns)  # Restore original feature names

# Save feature names before SMOTE
feature_names = X_train.columns

# Check class distribution before balancing (Training Data)
print("\nClass distribution in training data before balancing:", Counter(y_train))

# Apply SMOTE for oversampling (Training Data)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Convert X_train_balanced back to DataFrame with original column names
X_train_balanced = pd.DataFrame(X_train_balanced, columns=feature_names)

# Check class distribution after balancing (Training Data)
print("\nClass distribution in training data after SMOTE balancing:", Counter(y_train_balanced))

# Apply SMOTE to Testing Data 
print("\nWARNING: Applying SMOTE to testing data may distort evaluation metrics.")
print("This is not a standard practice and may lead to biased results.")

# Convert X_test back to a DataFrame if it's a NumPy array
if isinstance(X_test, np.ndarray):
    X_test = pd.DataFrame(X_test, columns=feature_names)

# Check class distribution before balancing (Testing Data)
print("\nClass distribution in testing data before balancing:", Counter(y_test))

# Apply SMOTE for oversampling (Testing Data)
X_test_balanced, y_test_balanced = smote.fit_resample(X_test, y_test)

# Convert X_test_balanced back to DataFrame with original column names
X_test_balanced = pd.DataFrame(X_test_balanced, columns=feature_names)

# Check class distribution after balancing (Testing Data)
print("\nClass distribution in testing data after SMOTE balancing:", Counter(y_test_balanced))

# Verify columns before transforming
print("\nColumns in X_train_balanced after SMOTE and feature selection:", X_train_balanced.columns)
print("\nColumns in X_test_balanced after SMOTE and feature selection:", X_test_balanced.columns)

# -----------------------------------------------------------------------------
# 2.5. Using Pipelines to streamline preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

selected_feature_names = X_train_balanced.columns  # Get the correct column names

# Define preprocessing steps for numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), selected_feature_names)  # Apply scaling only to selected features
    ]
)

# Create pipeline without feature selection (it was already applied)
pipeline = Pipeline([
    ("preprocessing", preprocessor)
])

# Fit-transform the training data
X_train_transformed = pipeline.fit_transform(X_train_balanced)

# Transform the test data
X_test_transformed = pipeline.transform(X_test)

print("\nPreprocessing pipeline applied successfully.")

print("\nShape of Transformed Training Data:", X_train_transformed.shape)
print("Shape of Transformed Test Data:", X_test_transformed.shape)
# Display first 5 rows (note: values are scaled, so they may look different)
print("\nPreview of Transformed Training Data:")
print(pd.DataFrame(X_train_transformed).head())
print("\nPreview of Transformed Test Data:")
print(pd.DataFrame(X_test_transformed).head())
