# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import io # Required for st.info buffer

# --- Configuration ---
st.set_page_config(page_title="Credit Risk Prediction", layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False) # Suppress Pyplot warning

# --- Caching Functions ---
@st.cache_data
def load_data(file_path):
    """Loads the CSV data, handling potential errors."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

@st.cache_data
def preprocess_data(df_raw):
    """Applies preprocessing steps: imputation, dropping columns, encoding, scaling."""
    df = df_raw.copy()

    # --- Imputation ---
    # Mean for roughly symmetric numeric columns
    mean_cols = ['age', 'credit_amount', 'duration', 'income', 'interest_rate']
    for col in mean_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)

    # Median for skewed numeric column
    median_cols = ['existing_loans_count']
    for col in median_cols:
         if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Mode for categorical columns
    mode_cols = ['job', 'marital_status', 'education_level', 'loan_type']
    for col in mode_cols:
         if col in df.columns:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    # --- Drop Columns ---
    df.drop(columns=['purpose', 'loan_eligibility'], inplace=True, errors='ignore')

    # --- Drop Rows with Remaining NaNs ---
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    rows_dropped = initial_rows - df.shape[0]

    # --- Encoding ---
    # Label Encode Target Variable ('default')
    if 'default' in df.columns:
        le_default = LabelEncoder()
        df['default'] = le_default.fit_transform(df['default'])

    # Label Encode 'sex'
    if 'sex' in df.columns:
        le_sex = LabelEncoder()
        df['sex'] = le_sex.fit_transform(df['sex'])

    # Separate Features (X) and Target (Y) before One-Hot Encoding
    if 'default' not in df.columns:
        st.error("Target column 'default' not found after initial processing.")
        return None, None, rows_dropped # Return None if target is missing

    X_raw = df.drop(columns=['default'])
    Y = df['default']

    # One-Hot Encode remaining categorical features
    X = pd.get_dummies(X_raw, drop_first=True).astype(int)

    # --- Scaling Numeric Features ---
    numeric_cols = ['age', 'credit_amount', 'duration', 'number_of_dependents', 'income',
                    'existing_loans_count', 'credit_history_length', 'previous_defaults',
                    'credit_score', 'installment_rate', 'interest_rate']
    # Ensure only columns present in X are scaled
    numeric_cols_in_x = [col for col in numeric_cols if col in X.columns]
    if numeric_cols_in_x:
        scaler = StandardScaler()
        X[numeric_cols_in_x] = scaler.fit_transform(X[numeric_cols_in_x])

    return X, Y, rows_dropped

# --- Model Training and Tuning ---
@st.cache_resource # Cache the trained model object
def train_initial_model(X_train, Y_train):
    """Trains the initial Logistic Regression model."""
    logreg = LogisticRegression(max_iter=2000, class_weight='balanced', solver='saga')
    logreg.fit(X_train, Y_train)
    return logreg

@st.cache_resource # Cache the grid search results and best model
def tune_hyperparameters(X_train, Y_train):
    """Performs GridSearchCV for Logistic Regression."""
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['saga'],
        'max_iter': [2000]
    }
    grid_search = GridSearchCV(estimator=LogisticRegression(class_weight='balanced'),
                               param_grid=param_grid,
                               scoring='roc_auc',
                               cv=5,
                               n_jobs=-1,
                               verbose=0) # Reduced verbosity for Streamlit
    grid_search.fit(X_train, Y_train)
    return grid_search

# --- Plotting Functions ---
def plot_confusion_matrix(y_true, y_pred, title):
    """Plots a confusion matrix using seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    st.pyplot(fig)

# --- Streamlit App ---
def main():
    st.title("📊 Credit Risk Prediction using Logistic Regression")
    st.markdown("This app demonstrates the process of building and evaluating a Logistic Regression model for credit risk prediction based on the provided dataset.")

    # --- 1. Load Data ---
    st.header("1. Data Loading")
    data_file = 'Credit_risk_data.csv'
    df_raw = load_data(data_file)

    if df_raw is not None:
        st.subheader("Raw Data Sample")
        st.dataframe(df_raw.head())
        st.write(f"Original Data Shape: `{df_raw.shape}`")

        # --- 2. Initial Data Exploration ---
        st.header("2. Initial Data Exploration")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Types")
            buffer = io.StringIO()
            df_raw.info(buf=buffer)
            st.text(buffer.getvalue())

            st.subheader("Missing Values (Before Imputation)")
            st.dataframe(df_raw.isna().sum().reset_index().rename(columns={0: 'Missing Count', 'index':'Column'}))

        with col2:
            st.subheader("Target Variable Distribution")
            fig, ax = plt.subplots()
            df_raw['default'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
            ax.set_title("Distribution of 'default' Target Variable")
            ax.set_xlabel("Default Status")
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=0)
            st.pyplot(fig)

            st.subheader("Numerical Feature Distributions (Histograms)")
            hist_cols = ["age","credit_amount","duration","income","existing_loans_count","interest_rate"]
            fig_hist, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten() # Flatten to 1D array for easier iteration
            for i, col in enumerate(hist_cols):
                 if col in df_raw.columns:
                    df_raw[[col]].hist(bins=20, edgecolor="black", ax=axes[i])
                    axes[i].set_title(col)
            plt.tight_layout()
            st.pyplot(fig_hist)


        # --- 3. Data Preprocessing ---
        st.header("3. Data Preprocessing")
        with st.spinner("Preprocessing data..."):
            X, Y, rows_dropped = preprocess_data(df_raw)

        if X is None or Y is None:
            st.stop() # Stop execution if preprocessing failed

        st.success("Preprocessing complete!")
        st.write(f"Rows dropped due to remaining NaNs after imputation: `{rows_dropped}`")
        st.write(f"Data Shape after Preprocessing (Features X): `{X.shape}`")
        st.write(f"Data Shape after Preprocessing (Target Y): `{Y.shape}`")

        st.subheader("Features after Encoding and Scaling (Sample)")
        st.dataframe(X.head())

        # --- 4. Train/Test Split ---
        st.header("4. Train/Test Split")
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y
        )
        st.write(f"Training Set Shape: X=`{X_train.shape}`, Y=`{Y_train.shape}`")
        st.write(f"Testing Set Shape: X=`{X_test.shape}`, Y=`{Y_test.shape}`")

        # --- 5. Initial Model Training & Evaluation ---
        st.header("5. Initial Logistic Regression Model")
        with st.spinner("Training initial model..."):
            logreg_initial = train_initial_model(X_train, Y_train)
        st.success("Initial model trained!")

        st.subheader("Initial Model Evaluation (on Test Set)")
        Y_pred_initial = logreg_initial.predict(X_test)
        Y_proba_initial = logreg_initial.predict_proba(X_test)[:, 1]

        acc_initial = accuracy_score(Y_test, Y_pred_initial)
        roc_auc_initial = roc_auc_score(Y_test, Y_proba_initial)
        report_initial = classification_report(Y_test, Y_pred_initial, digits=3)

        st.metric("Accuracy", f"{acc_initial:.3f}")
        st.metric("ROC AUC Score", f"{roc_auc_initial:.3f}")
        st.subheader("Classification Report")
        st.text(report_initial)
        plot_confusion_matrix(Y_test, Y_pred_initial, "Initial Model Confusion Matrix")

        st.divider()

        # --- 6. Hyperparameter Tuning ---
        st.header("6. Hyperparameter Tuning (GridSearchCV)")
        with st.spinner("Performing Grid Search... (This might take a moment)"):
            grid_search = tune_hyperparameters(X_train, Y_train)
        st.success("Grid Search complete!")

        st.subheader("Best Parameters Found")
        st.write(grid_search.best_params_)
        best_logreg = grid_search.best_estimator_

        # --- 7. Tuned Model Evaluation ---
        st.header("7. Tuned Logistic Regression Model Evaluation")
        st.subheader("Tuned Model Evaluation (on Test Set)")
        Y_pred_tuned = best_logreg.predict(X_test)
        Y_proba_tuned = best_logreg.predict_proba(X_test)[:, 1]

        acc_tuned = accuracy_score(Y_test, Y_pred_tuned)
        roc_auc_tuned = roc_auc_score(Y_test, Y_proba_tuned)
        report_tuned = classification_report(Y_test, Y_pred_tuned, digits=3)

        st.metric("Tuned Model Accuracy", f"{acc_tuned:.3f}")
        st.metric("Tuned Model ROC AUC Score", f"{roc_auc_tuned:.3f}")
        st.subheader("Tuned Model Classification Report")
        st.text(report_tuned)
        plot_confusion_matrix(Y_test, Y_pred_tuned, "Tuned Model Confusion Matrix")

        st.divider()

        # --- 8. Generalization Check ---
        st.header("8. Model Generalization Check")
        Y_pred_train_tuned = best_logreg.predict(X_train)
        train_accuracy_tuned = accuracy_score(Y_train, Y_pred_train_tuned)
        test_accuracy_tuned = accuracy_score(Y_test, Y_pred_tuned) # Same as acc_tuned

        st.write(f"Tuned Model Train Accuracy: `{train_accuracy_tuned:.3f}`")
        st.write(f"Tuned Model Test Accuracy: `{test_accuracy_tuned:.3f}`")

        # Determine generalization status based on the tuned model
        if abs(train_accuracy_tuned - test_accuracy_tuned) < 0.05:
            st.success("✅ Good generalization (Train and Test accuracies are similar)")
        elif train_accuracy_tuned > test_accuracy_tuned + 0.1:
            st.warning("⚠️ Overfitting may be present (Train accuracy significantly higher than Test).")
            st.markdown("""
                **Potential Solutions:**
                * Use stronger regularization (adjust 'C' parameter).
                * Perform feature selection/engineering.
                * Gather more diverse training data.
            """)
        else: # Covers underfitting or other unexpected gaps
            st.info("❓ Check model complexity or data representativeness (potential underfitting or other issues).")

if __name__ == "__main__":
    main()
