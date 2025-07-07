# Heart Attack Prediction with Logistic Regression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- 1. Load Dataset ---
file_path = "heart_attack_prediction_dataset.csv"
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully. First 5 rows:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct directory.")
    exit()

# --- 2. Dataset Exploration ---
print("\n--- Dataset Exploration ---")
print("\nData Info:")
df.info()

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nHeart Attack Risk distribution:")
print(df['Heart Attack Risk'].value_counts(normalize=True))
print("This indicates a class imbalance.")


# --- 3. Data Preprocessing ---
print("\n--- Data Preprocessing ---")

# Drop non-predictive ID and high-cardinality location columns
columns_to_drop = ["Patient ID", "Country", "Continent", "Hemisphere"]
df = df.drop(columns=columns_to_drop, axis=1)
print(f"Dropped columns: {', '.join(columns_to_drop)}")

# Convert Blood Pressure from 'SYS/DIA' string to two numerical columns
print("Processing 'Blood Pressure' column...")
bp_split = df["Blood Pressure"].str.split("/", expand=True)
df["Systolic_BP"] = pd.to_numeric(bp_split[0], errors='coerce')
df["Diastolic_BP"] = pd.to_numeric(bp_split[1], errors='coerce')
df.drop("Blood Pressure", axis=1, inplace=True)

# Handle any NaNs that might have been introduced (robustness check)
if df[["Systolic_BP", "Diastolic_BP"]].isnull().sum().any():
    df.dropna(subset=["Systolic_BP", "Diastolic_BP"], inplace=True)

# One-hot encode remaining categorical features
print("Encoding categorical features (Sex, Diet)...")
df = pd.get_dummies(df, columns=['Sex', 'Diet'], drop_first=True)


# --- 4. Define Features (X) and Target (y) ---
X = df.drop("Heart Attack Risk", axis=1)
y = df["Heart Attack Risk"]
print(f"Number of features after preprocessing: {X.shape[1]}")


# --- 5. Train-Test Split ---
# WHY: To evaluate the model on unseen data. Stratification is crucial for imbalanced datasets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")


# --- 6. Create Preprocessing and Modeling Pipeline ---
print("\n--- Model Training Pipeline ---")

# WHY: A Pipeline encapsulates preprocessing and modeling. It standardizes data correctly
# by fitting the scaler ONLY on the training data and then transforming both train and test sets,
# preventing data leakage and simplifying the workflow.

# Since all features are now numeric, we create a simple preprocessor to scale them all.
numeric_features = X.columns
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numeric_features)],
    remainder='passthrough'
)

# Define the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver="liblinear", random_state=42, class_weight='balanced', max_iter=1000))
])

# Train the entire pipeline on the training data
model_pipeline.fit(X_train, y_train)
print("Model pipeline trained successfully.")


# --- 7. Predictions ---
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]


# --- 8. Evaluation ---
print("\n--- Model Evaluation ---")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Class 1): {precision:.4f}")
print(f"Recall (Class 1): {recall:.4f}")
print(f"F1 Score (Class 1): {f1:.4f}")
print(f"AUC Score: {auc_score:.4f}")

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_lr.png")
plt.close()

# Plot ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_lr.png")
plt.close()


# --- 9. Feature Importance ---
print("\n--- Feature Importance (Model Coefficients) ---")
# WHY: Coefficients show the strength and direction of each feature's relationship with the outcome.
coefficients = model_pipeline.named_steps['classifier'].coef_[0]
feature_names = X.columns # Column names from the original dataframe

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", key=abs, ascending=False)

print(feature_importance)

# Plot top N most important features
plt.figure(figsize=(10, 8))
top_n = 20
sns.barplot(x="Coefficient", y="Feature", data=feature_importance.head(top_n), palette='viridis')
plt.title(f"Top {top_n} Feature Importances (Logistic Regression Coefficients)")
plt.tight_layout()
plt.savefig("feature_importance_plot_lr.png")
plt.close()

print("\nCode execution complete. Check generated plots (confusion_matrix_lr.png, roc_curve_lr.png, feature_importance_plot_lr.png).")