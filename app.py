# app.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# ============================
# 1. Load Dataset
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("depression_student_dataset.csv")
    df = df.dropna()
    return df

df = load_data()

st.title("Student Depression Prediction")
st.write("### Dataset Overview")
st.dataframe(df.head())

# ============================
# 2. Preprocessing
# ============================
# Identify categorical columns (exclude target)
categorical_cols = [col for col in df.select_dtypes(include=['object']).columns if col != "Depression"]

# Encode categorical feature columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Convert target column to numeric (0 = No, 1 = Yes)
df['Depression'] = df['Depression'].map({'No': 0, 'Yes': 1})

X = df.drop("Depression", axis=1)
y = df["Depression"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# 3. Train Model
# ============================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

# ============================
# 4. Model Performance
# ============================
st.write("### Model Accuracy")
st.write(accuracy_score(y_test, y_pred))

st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.write("### ROC Curve")
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, color="darkorange", label="ROC curve (area = %0.2f)" % roc_auc)
ax2.plot([0, 1], [0, 1], color="navy", linestyle="--")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve - Random Forest")
ax2.legend(loc="lower right")
st.pyplot(fig2)

# ============================
# 5. User Input for Prediction
# ============================
st.write("### Predict Depression for a New Student")

def user_input_features():
    data = {}
    for col in X.columns:
        if col in categorical_cols:
            le = encoders[col]
            unique_vals = df[col].unique()
            original_labels = le.inverse_transform(unique_vals)
            selected_label = st.selectbox(f"{col}", original_labels)
            # Convert back to encoded value
            data[col] = le.transform([selected_label])[0]
        else:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            data[col] = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Scale input
input_scaled = scaler.transform(input_df)
prediction = rf.predict(input_scaled)
prediction_proba = rf.predict_proba(input_scaled)

st.write("### Prediction")
st.write("Depression" if prediction[0] == 1 else "No Depression")
st.write("Prediction Probability:", prediction_proba[0])
