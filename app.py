import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib # Using joblib for potentially saving/loading model later if needed

# --- Configuration ---
st.set_page_config(page_title="Credit Risk Predictor", layout="wide")
st.title("Credit Risk Prediction App")
st.write("Enter customer details to predict credit default risk.")

# --- Global Variables & Constants ---
DATA_PATH = 'Credit_risk_data.csv'
TARGET_COLUMN = 'default'
# Best parameters found in the notebook's grid search
BEST_DT_PARAMS = {
    'criterion': 'entropy',
    'max_depth': 10,
    'min_samples_leaf': 2,
    'min_samples_split': 20,
    'class_weight': 'balanced',
    'random_state': 42
}

# --- Caching Data Loading and Preprocessing ---
# Cache the data loading and initial preprocessing steps
# This prevents reloading and reprocessing on every interaction
@st.cache_data
def load_and_preprocess_data(file_path):
    """Loads and preprocesses the credit risk data."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The data file '{file_path}' was not found. Please place it in the same directory as app.py.")
        return None, None, None, None, None, None, None

    # --- Preprocessing Steps (as in the notebook) ---

    # Impute missing numeric values
    numeric_means = {
        'age': df['age'].mean(),
        'credit_amount': df['credit_amount'].mean(),
        'duration': df['duration'].mean(),
        'income': df['income'].mean(),
        'interest_rate': df['interest_rate'].mean()
    }
    numeric_medians = {
        'existing_loans_count': df['existing_loans_count'].median()
    }
    for col, mean_val in numeric_means.items():
        df[col] = df[col].fillna(mean_val)
    for col, median_val in numeric_medians.items():
        df[col] = df[col].fillna(median_val)

    # Impute missing categorical values with mode
    categorical_modes = {
        'job': df['job'].mode()[0],
        'marital_status': df['marital_status'].mode()[0],
        'education_level': df['education_level'].mode()[0],
        'loan_type': df['loan_type'].mode()[0],
        # Imputing other categoricals that were previously dropped rows for
        'sex': df['sex'].mode()[0],
        'saving_accounts': df['saving_accounts'].mode()[0],
        'checking_account': df['checking_account'].mode()[0],
        'employment_status': df['employment_status'].mode()[0],
        'collateral': df['collateral'].mode()[0]
    }
    for col, mode_val in categorical_modes.items():
        df[col] = df[col].fillna(mode_val)

    # Impute remaining numeric columns (previously dropped rows)
    # Using median for potentially skewed data
    remaining_numeric_medians = {
        'number_of_dependents': df['number_of_dependents'].median(),
        'credit_history_length': df['credit_history_length'].median(),
        'previous_defaults': df['previous_defaults'].median(),
        'credit_score': df['credit_score'].median(),
        'installment_rate': df['installment_rate'].median(),
    }
    for col, median_val in remaining_numeric_medians.items():
        df[col] = df[col].fillna(median_val)


    # Drop unnecessary columns
    df = df.drop(columns=['purpose', 'loan_eligibility'], errors='ignore')

    # Ensure no NaNs remain before encoding/scaling (important after changing NaN strategy)
    if df.isnull().sum().sum() > 0:
        st.warning("Warning: Some NaNs remained after imputation. Dropping rows with NaNs.")
        df = df.dropna()

    if df.empty:
        st.error("Error: DataFrame became empty after handling missing values.")
        return None, None, None, None, None, None, None


    # --- Encoding ---
    # Label Encode Binary Target and Feature
    label_encoders = {}
    if TARGET_COLUMN in df.columns:
        le_target = LabelEncoder()
        df[TARGET_COLUMN] = le_target.fit_transform(df[TARGET_COLUMN])
        label_encoders[TARGET_COLUMN] = le_target

    if 'sex' in df.columns:
        le_sex = LabelEncoder()
        df['sex'] = le_sex.fit_transform(df['sex'])
        label_encoders['sex'] = le_sex

    # Separate Features (X) and Target (Y) BEFORE one-hot encoding Y
    X = df.drop(columns=[TARGET_COLUMN])
    Y = df[TARGET_COLUMN]

    # One-Hot Encode remaining categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
    processed_cols = X.columns.tolist() # Store column names after OHE

    # Store values needed for processing new input
    preprocessing_info = {
        'numeric_means': numeric_means,
        'numeric_medians': numeric_medians,
        'categorical_modes': categorical_modes,
        'remaining_numeric_medians': remaining_numeric_medians,
        'label_encoders': label_encoders,
        'processed_columns': processed_cols,
        'original_categorical_cols': categorical_cols.tolist()
    }

    return X, Y, preprocessing_info, df # Return original df for input options

# --- Model Training (Cached) ---
@st.cache_resource # Cache the trained model object
def train_model(X_train, Y_train, X_test, Y_test):
    """Trains the Decision Tree model and fits the scaler."""
    # --- Scaling (Fit ONLY on Training Data) ---
    numeric_cols_to_scale = ['age', 'credit_amount', 'duration', 'number_of_dependents', 'income',
                             'existing_loans_count', 'credit_history_length', 'previous_defaults',
                             'credit_score', 'installment_rate', 'interest_rate']
    # Ensure only columns present in X_train are scaled
    numeric_cols_present = [col for col in numeric_cols_to_scale if col in X_train.columns]

    scaler = StandardScaler()
    # Fit scaler on training data and transform train/test
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols_present] = scaler.fit_transform(X_train[numeric_cols_present])
    X_test_scaled[numeric_cols_present] = scaler.transform(X_test[numeric_cols_present]) # Use transform here

    # --- Train Best Decision Tree Model ---
    dtree_best = DecisionTreeClassifier(**BEST_DT_PARAMS)
    dtree_best.fit(X_train_scaled, Y_train)

    # --- Evaluate ---
    Y_pred_best = dtree_best.predict(X_test_scaled)
    accuracy = accuracy_score(Y_test, Y_pred_best)
    try:
        roc_auc = roc_auc_score(Y_test, dtree_best.predict_proba(X_test_scaled)[:, 1])
    except Exception as e:
        roc_auc = f"Could not calculate ROC AUC: {e}" # Handle potential errors if only one class in test set


    return dtree_best, scaler, accuracy, roc_auc, numeric_cols_present

# --- Load Data and Train Model ---
X, Y, preprocessing_info, df_original = load_and_preprocess_data(DATA_PATH)

model = None
scaler = None
accuracy = "N/A"
roc_auc = "N/A"
scaled_numeric_cols = []

if X is not None and Y is not None:
    # Split data AFTER initial NaN handling and encoding
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    # Train model and fit scaler
    model, scaler, accuracy, roc_auc, scaled_numeric_cols = train_model(X_train, Y_train, X_test, Y_test)

    # Display Model Performance
    st.sidebar.subheader("Model Performance (on Test Set)")
    st.sidebar.write(f"Accuracy: {accuracy:.3f}")
    st.sidebar.write(f"ROC AUC: {roc_auc:.3f}" if isinstance(roc_auc, float) else f"ROC AUC: {roc_auc}")

else:
    st.stop() # Stop execution if data loading failed

# --- User Input ---
st.sidebar.header("Enter Customer Details:")

input_data = {}

# Create input fields based on original columns before dummification
original_cols = df_original.drop(columns=[TARGET_COLUMN, 'purpose', 'loan_eligibility'], errors='ignore').columns

# Determine unique values for selectboxes BEFORE dropping NaNs
unique_values = {col: df_original[col].unique().tolist() for col in preprocessing_info['original_categorical_cols'] if col != 'sex'}

for col in original_cols:
    if col in scaled_numeric_cols:
        # Use median as default for numeric inputs
        default_val = preprocessing_info['numeric_means'].get(col,
                         preprocessing_info['numeric_medians'].get(col,
                         preprocessing_info['remaining_numeric_medians'].get(col, 0))) # Fallback to 0 if missing (shouldn't happen)
        # Ensure default is float for number_input
        input_data[col] = st.sidebar.number_input(f"Enter {col.replace('_', ' ').title()}", value=float(default_val))
    elif col == 'sex':
        # Use mode as default index
        default_sex_str = preprocessing_info['categorical_modes']['sex']
        sex_options = ['female', 'male'] # Based on typical encoding
        try:
             # Ensure default_sex_str is in options before finding index
            if default_sex_str not in sex_options:
                default_sex_str = sex_options[0] # Fallback if mode is unexpected
            default_sex_index = sex_options.index(default_sex_str)
        except ValueError:
            default_sex_index = 0 # Fallback index
        input_data[col] = st.sidebar.selectbox(f"Select {col.replace('_', ' ').title()}", options=sex_options, index=default_sex_index)
    elif col in preprocessing_info['original_categorical_cols']:
        # Use mode as default index for other categoricals
        options = unique_values.get(col, [preprocessing_info['categorical_modes'].get(col, "N/A")])
         # Convert options to string to ensure consistency
        options = [str(opt) for opt in options]
        default_cat_str = str(preprocessing_info['categorical_modes'].get(col, options[0] if options else "N/A"))
        try:
            # Ensure default_cat_str is in options before finding index
            if default_cat_str not in options:
                 default_cat_str = options[0] if options else "N/A" # Fallback if mode is unexpected
            default_cat_index = options.index(default_cat_str)
        except ValueError:
            default_cat_index = 0 # Fallback index
        input_data[col] = st.sidebar.selectbox(f"Select {col.replace('_', ' ').title()}", options=options, index=default_cat_index)
    # Skip columns that were dropped or are the target

# --- Prediction Logic ---
def preprocess_input(input_dict, info, scaler_obj, numeric_cols):
    """Preprocesses a single dictionary of user input."""
    input_df = pd.DataFrame([input_dict])

    # Impute potential missing inputs (though widgets should have defaults) - using training set values
    for col, mean_val in info['numeric_means'].items():
        if col in input_df.columns: input_df[col] = input_df[col].fillna(mean_val)
    for col, median_val in info['numeric_medians'].items():
        if col in input_df.columns: input_df[col] = input_df[col].fillna(median_val)
    for col, mode_val in info['categorical_modes'].items():
        if col in input_df.columns: input_df[col] = input_df[col].fillna(mode_val)
    for col, median_val in info['remaining_numeric_medians'].items():
         if col in input_df.columns: input_df[col] = input_df[col].fillna(median_val)

    # Label Encode 'sex'
    if 'sex' in input_df.columns and 'sex' in info['label_encoders']:
        le_sex = info['label_encoders']['sex']
        # Handle unseen values during transform gracefully
        input_df['sex'] = input_df['sex'].apply(lambda x: le_sex.transform([x])[0] if x in le_sex.classes_ else -1) # Use -1 or another indicator for unseen

    # One-Hot Encode other categoricals
    input_df = pd.get_dummies(input_df, columns=info['original_categorical_cols'], drop_first=True, dtype=int)

    # Align columns with the training data (handles missing/extra columns from dummification)
    input_df = input_df.reindex(columns=info['processed_columns'], fill_value=0)

    # Scale numeric features
    numeric_cols_present_input = [col for col in numeric_cols if col in input_df.columns]
    if numeric_cols_present_input: # Check if there are numeric columns to scale
        input_df[numeric_cols_present_input] = scaler_obj.transform(input_df[numeric_cols_present_input])

    return input_df


# --- Predict Button and Output ---
if st.sidebar.button("Predict Credit Risk"):
    if model and scaler and preprocessing_info:
        # Preprocess the user input
        processed_input = preprocess_input(input_data, preprocessing_info, scaler, scaled_numeric_cols)

        # Make prediction
        prediction = model.predict(processed_input)[0]
        prediction_proba = model.predict_proba(processed_input)[0]

        # Map prediction back to labels
        le_target = preprocessing_info['label_encoders'][TARGET_COLUMN]
        predicted_label = le_target.inverse_transform([prediction])[0]

        # Display result
        st.subheader("Prediction Result:")
        if predicted_label == 'bad':
            st.error("Predicted Risk: Bad Credit Risk")
        else:
            st.success("Predicted Risk: Good Credit Risk")

        st.write("Prediction Probabilities:")
        st.write(f"- Probability of Good Credit Risk: {prediction_proba[le_target.transform(['good'])[0]]:.2f}")
        st.write(f"- Probability of Bad Credit Risk: {prediction_proba[le_target.transform(['bad'])[0]]:.2f}")

        # st.write("Processed Input for Model (scaled):")
        # st.dataframe(processed_input) # Optional: show the final input to the model

    else:
        st.error("Model or preprocessing data not available. Cannot predict.")
