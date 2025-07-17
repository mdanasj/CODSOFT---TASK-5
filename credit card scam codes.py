# 📦 Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 🗂️ Load the dataset
df = pd.read_csv("creditcard.csv.csv")  # Replace with your path if needed

# 🧼 Normalize 'Amount' and 'Time'
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])
df["Time"] = scaler.fit_transform(df[["Time"]])

# 🎯 Features & Target
X = df.drop("Class", axis=1)
y = df["Class"]

# ⚖️ Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 🔪 Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# 🧠 Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔍 Evaluate performance
y_pred = model.predict(X_test)
print("\n🔔 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
print("✅ Model trained — evaluating now...")
input("Press Enter to exit...")
