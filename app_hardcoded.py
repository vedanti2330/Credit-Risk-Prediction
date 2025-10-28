
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

st.set_page_config(page_title="Credit Risk Prediction (Logistic)", page_icon="📊", layout="wide")
st.title("📊 Credit Risk Prediction — Logistic Regression")

DATASET_PATH = 'Credit_risk_data.csv'
TARGET_COL = 'default'
st.caption(f"Dataset: **{DATASET_PATH}**, Target: **{TARGET_COL}** (auto-detected)")

def load_and_prepare(data_path: str, target: str):
    df = pd.read_csv(data_path)
    df = df.copy().drop_duplicates()
    if df[target].dtype == 'object':
        mapped = (
            df[target].astype(str).str.strip().str.lower()
            .map({'yes':1,'y':1,'true':1,'no':0,'n':0,'false':0})
        )
        df[target] = mapped.where(mapped.notna(), df[target])
    try:
        df[target] = pd.to_numeric(df[target])
    except Exception:
        pass

    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    if cat_cols:
        X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])
    if num_cols:
        X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])

    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    if num_cols:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y, num_cols

if st.button("🔁 Train / Refresh"):
    st.experimental_rerun()

try:
    X, y, num_cols = load_and_prepare(DATASET_PATH, TARGET_COL)
except FileNotFoundError:
    st.error(f"Could not find dataset at '{DATASET_PATH}'. Place the CSV next to app.py or adjust DATASET_PATH in code.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load/prepare data: {e}")
    st.stop()

st.subheader("Preview")
st.dataframe(pd.read_csv(DATASET_PATH).head())

stratify_y = y if y.nunique() <= 20 else None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_y)

model = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
try:
    y_prob = model.predict_proba(X_test)[:,1]
except Exception:
    y_prob = None

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

from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
st.subheader("Confusion Matrix")
st.write(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))
fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay(cm, display_labels=[0,1]).plot(ax=ax_cm)
st.pyplot(fig_cm)

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

st.subheader("🔮 Single Prediction")
with st.form("predict_form"):
    user_values = {}
    for col in X.columns:
        if col in num_cols:
            user_values[col] = st.number_input(col, value=float(np.nan))
        else:
            user_values[col] = st.selectbox(col, [0,1], index=0)
    do_predict = st.form_submit_button("Predict")
    if do_predict:
        ui = pd.DataFrame([user_values]).reindex(columns=X.columns, fill_value=0)
        try:
            prob = model.predict_proba(ui)[:,1][0]
            pred = int(prob >= 0.5)
            st.success(f"Prediction: **{pred}**  |  Probability of class 1: **{prob:.3f}**")
        except Exception as e:
            try:
                pred = int(model.predict(ui)[0])
                st.success(f"Prediction: **{pred}**")
            except Exception as e2:
                st.error(f"Could not predict: {e2}")
