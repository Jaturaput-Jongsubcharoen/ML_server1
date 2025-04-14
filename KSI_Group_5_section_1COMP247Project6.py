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
# 2.2. Feature preprocessing (numerical + categorical)  
# Separate features and target
y = Group_data["ACCLASS"].apply(lambda x: 1 if x == "Fatal" else 0)
X = Group_data.drop(columns=["ACCLASS"])

# Define categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Define preprocessing for numerical data
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Define preprocessing for categorical data
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Store final feature names for later (optional)
encoded_cat_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
final_feature_names = numerical_cols + list(encoded_cat_names)

# Optional: show dimensions
print(f"\nCombined features shape: {X_processed.shape}")

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2.3. Train, Test data splitting – Using train_test_split (Apply SMOTE)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2.4. Managing imbalanced classes – Oversampling / Undersampling
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd

print("\nClass distribution before SMOTE:", Counter(y_train))

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE:", Counter(y_train_balanced))

# Optional: Convert to DataFrame for inspection
X_train_balanced = pd.DataFrame(X_train_balanced, columns=final_feature_names)
X_test = pd.DataFrame(X_test, columns=final_feature_names)
# -----------------------------------------------------------------------------

# 2.5. Using Pipelines to streamline preprocessing (AFTER SMOTE)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Separate numerical and categorical columns from the full original dataset
# This ensures the pipeline uses correct column types (after SMOTE)
numerical_cols = Group_data.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = Group_data.select_dtypes(include=["object"]).columns.tolist()

# Rebuild X_train_balanced and X_test using the original column structure
# Important: Use the same columns as the original data before encoding
X_full = Group_data.drop(columns=["ACCLASS"], errors="ignore")
y = Group_data["ACCLASS"].apply(lambda x: 1 if x == "Fatal" else 0)

# Train-test split (before SMOTE)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE after encoding (only on training data)
from imblearn.over_sampling import SMOTE
from collections import Counter

# Define preprocessing steps
numerical_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # sparse_output=False for compatibility
])

# Combine steps
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_preprocessor, numerical_cols),
    ("cat", categorical_preprocessor, categorical_cols)
])

# Apply preprocessing to training data
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Get feature names after encoding
encoded_cat_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
final_feature_names = numerical_cols + list(encoded_cat_names)

# Apply SMOTE on encoded training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_encoded, y_train)

# Preview outputs
import pandas as pd
X_train_transformed = pd.DataFrame(X_train_balanced, columns=final_feature_names)
X_test_transformed = pd.DataFrame(X_test_encoded, columns=final_feature_names)

print("\nPreprocessing pipeline applied successfully.")
print("Shape of Transformed Training Data:", X_train_transformed.shape)
print("Shape of Transformed Test Data:", X_test_transformed.shape)
print("\nPreview of Transformed Training Data:")
print(X_train_transformed.head())


# -----------------------------------------------------------------------------  
# 3. Predictive model building  
# -----------------------------------------------------------------------------  
# -----------------------------------------------------------------------------
# 3.1 Predictive Model Building - KNN (with Transformed Features)
# -----------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train KNN model using transformed training data
knn_model.fit(X_train_transformed, y_train_balanced)

# Predict using the transformed test data
y_pred_knn = knn_model.predict(X_test_transformed)

# Evaluate KNN model
acc = accuracy_score(y_test, y_pred_knn)
print("\nKNN - Accuracy: {:.4f}".format(acc))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("Classification Report:")
print(classification_report(y_test, y_pred_knn))

# -----------------------------------------------------------------------------
# 3.2 Predictive Model Building - Random Forest (with Transformed Features)
# -----------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=25)

# Train the model using transformed, balanced training data
rf_model.fit(X_train_transformed, y_train_balanced)

# Predict using the transformed test data
y_pred_rf = rf_model.predict(X_test_transformed)

# Evaluate the model
acc_rf = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest - Accuracy: {:.4f}".format(acc_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

# -----------------------------------------------------------------------------
# 3.3 Predictive Model Building - SVM (with Transformed Features)
# -----------------------------------------------------------------------------
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the SVM model with probability output enabled
svm_model = SVC(probability=True, random_state=25)

# Train the model using transformed, balanced training data
svm_model.fit(X_train_transformed, y_train_balanced)

# Predict using the transformed test data
y_pred_svm = svm_model.predict(X_test_transformed)

# Evaluate the model
acc_svm = accuracy_score(y_test, y_pred_svm)
print("\nSVM - Accuracy: {:.4f}".format(acc_svm))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
print("Classification Report:")
print(classification_report(y_test, y_pred_svm))

# -----------------------------------------------------------------------------
# 3.4 Predictive Model Building - Logistic Regression (with Transformed Features)
# -----------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the Logistic Regression model
logreg_model = LogisticRegression(max_iter=1000, random_state=25)

# Train the model using transformed, balanced training data
logreg_model.fit(X_train_transformed, y_train_balanced)

# Predict using the transformed test data
y_pred_logreg = logreg_model.predict(X_test_transformed)

# Evaluate the model
acc_logreg = accuracy_score(y_test, y_pred_logreg)
print("\nLogistic Regression - Accuracy: {:.4f}".format(acc_logreg))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logreg))
print("Classification Report:")
print(classification_report(y_test, y_pred_logreg))

# -----------------------------------------------------------------------------
# 3.5 Predictive Model Building - Neural Network (MLPClassifier) with Transformed Features
# -----------------------------------------------------------------------------
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the Neural Network model
mlp_model = MLPClassifier(max_iter=1000, random_state=25)

# Train the model using transformed, balanced training data
mlp_model.fit(X_train_transformed, y_train_balanced)

# Predict using the transformed test data
y_pred_mlp = mlp_model.predict(X_test_transformed)

# Evaluate the model
acc_mlp = accuracy_score(y_test, y_pred_mlp)
print("\nNeural Network - Accuracy: {:.4f}".format(acc_mlp))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_mlp))
print("Classification Report:")
print(classification_report(y_test, y_pred_mlp))


# -----------------------------------------------------------------------------
# 4. Model scoring and evaluation
# -----------------------------------------------------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

# 4.1 Present results using accuracy, precision, recall, F1 score, and ROC curves
model_scores = {}

# Prepare models dictionary manually
model_info = {
    "KNN": knn_model,
    "Random Forest": rf_model,
    "SVM": svm_model,
    "Logistic Regression": logreg_model,
    "Neural Network": mlp_model
}

plt.figure(figsize=(8, 6))  # Setup for ROC plot

for name, model in model_info.items():
    y_pred = model.predict(X_test)

    # Predict probabilities or decision scores
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred  # fallback

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    model_scores[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }

    print("\n")
    print("----------------------------------------")
    print(f"{name}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("----------------------------------------")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

# Finalize ROC plot
plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.50)")  # Baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()
print("----------------------------------------")

# -----------------------------------------------------------------------------
# 4.2 Select and Recommend the Best Performing Model
# -----------------------------------------------------------------------------
results_df = pd.DataFrame(model_scores).T.sort_values("F1 Score", ascending=False)
print("\n")
print("----------------------------------------")
print("Model Performance Summary:")
print(results_df)
print("----------------------------------------")

best_model = results_df.index[0]
print("\n")
print("----------------------------------------")
print(f"Recommended Model: {best_model} (Based on highest F1 Score)")
print("----------------------------------------")

# -----------------------------------------------------------------------------
# 5. Save all trained models
# -----------------------------------------------------------------------------
import joblib
print("\n")
print("----------------------------------------")
joblib.dump(knn_model, "knn_model.pkl")
print("KNN model saved as 'knn_model.pkl'")

joblib.dump(rf_model, "random_forest_model.pkl")
print("Random Forest model saved as 'random_forest_model.pkl'")

joblib.dump(svm_model, "svm_model.pkl")
print("SVM model saved as 'svm_model.pkl'")

joblib.dump(logreg_model, "logistic_regression_model.pkl")
print("Logistic Regression model saved as 'logistic_regression_model.pkl'")

joblib.dump(mlp_model, "neural_network_model.pkl")
print("Neural Network model saved as 'neural_network_model.pkl'")
print("----------------------------------------")



# print("Top 5 STREET1 values used the most:")
# print(Group_data["STREET1"].value_counts().head(5))

# print("Top 5 STREET2 values used the most:")
# print(Group_data["STREET2"].value_counts().head(5))

# print("Top 5 OFFSET values used the most:")
# print(Group_data["OFFSET"].value_counts().head(5))