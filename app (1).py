
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression

# === Auto-detected from your notebook ===
DATASET_HINT = 'Credit_risk_data.csv'
TARGET_HINT = 'default'

st.set_page_config(page_title="Credit Risk Prediction (Logistic)", page_icon="📊", layout="wide")
st.title("📊 Credit Risk Prediction — Logistic Regression")

st.caption("This app was generated from your notebook. Upload your CSV or toggle the built-in path if detected.")

with st.expander("Notebook parsing summary"):
    st.write({'dataset_hint': 'Credit_risk_data.csv', 'target_hint': 'default'})

# Sidebar
st.sidebar.header("Data & Target")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
target_col = st.sidebar.text_input("Target column (0/1)", value=TARGET_HINT or "default")
use_default = False
if DATASET_HINT:
    use_default = st.sidebar.checkbox(f"Use notebook dataset path: {DATASET_HINT}", value=False)

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

st.sidebar.header("Model Params")
C = st.sidebar.number_input("C (inverse regularization)", value=1.0)
max_iter = st.sidebar.number_input("max_iter", value=200)
solver = st.sidebar.selectbox("solver", ["lbfgs","liblinear","newton-cg","saga","sag"], index=0)

# Load data
df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
elif use_default and DATASET_HINT:
    try:
        df = pd.read_csv(DATASET_HINT)
    except Exception as e:
        st.warning(f"Could not load default dataset: {e}")

if df is None:
    st.info("👆 Upload a CSV to proceed.")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head())

if target_col not in df.columns:
    st.error(f"Target '{target_col}' not found. Available columns: {list(df.columns)}")
    st.stop()

# Basic cleaning
df = df.copy()
df.drop_duplicates(inplace=True)

# Try converting common yes/no target patterns to 0/1
if df[target_col].dtype == 'object':
    df[target_col] = (
        df[target_col].astype(str).str.strip().str.lower()
        .map({'yes':1,'y':1,'true':1,'no':0,'n':0,'false':0})
        .fillna(np.nan)
    )
# Coerce target to numeric if possible
try:
    df[target_col] = pd.to_numeric(df[target_col])
except Exception:
    pass

# Drop rows with missing target
df = df.dropna(subset=[target_col])

# Features / Target
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify types
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

st.write("**Numeric features:**", num_cols)
st.write("**Categorical features:**", cat_cols)

# Impute and encode
if cat_cols:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
if num_cols:
    num_imputer = SimpleImputer(strategy="median")
    X[num_cols] = num_imputer.fit_transform(X[num_cols])

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Scale numeric
if num_cols:
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

# Split
stratify_y = y if y.nunique() <= 20 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
)

# Model
model = LogisticRegression(C=C, max_iter=int(max_iter), solver=solver)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
try:
    y_prob = model.predict_proba(X_test)[:,1]
except Exception:
    y_prob = None

# Metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
c2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
c3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.3f}")
c4.metric("F1", f"{f1_score(y_test, y_pred, zero_division=0):.3f}")
if y_prob is not None:
    try:
        auc = roc_auc_score(y_test, y_prob)
        c5.metric("ROC AUC", f"{auc:.3f}")
    except Exception:
        c5.metric("ROC AUC", "n/a")

# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
st.subheader("Confusion Matrix")
st.write(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))
fig_cm, ax_cm = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot(ax=ax_cm)
st.pyplot(fig_cm)

# ROC Curve
if y_prob is not None:
    fpr, tpr, thr = roc_curve(y_test, y_prob)
    st.subheader("ROC Curve")
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label="ROC")
    ax_roc.plot([0,1], [0,1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    st.pyplot(fig_roc)

# Inference form
st.subheader("🔮 Single Prediction")
with st.form("predict_form"):
    user_values = {}
    for col in X.columns:
        if col in num_cols:
            val = st.number_input(col, value=float(np.nan))
        else:
            val = st.selectbox(col, [0,1], index=0)
        user_values[col] = val
    submitted = st.form_submit_button("Predict")
    if submitted:
        ui = pd.DataFrame([user_values])
        # align columns
        ui = ui.reindex(columns=X.columns, fill_value=0)
        try:
            prob = model.predict_proba(ui)[:,1][0]
            pred = int(prob >= 0.5)
            st.success(f"Prediction: **{pred}**  |  Probability of class 1: **{prob:.3f}**")
        except Exception as e:
            st.error(f"Could not predict: {e}")
