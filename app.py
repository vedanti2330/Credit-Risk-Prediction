import pandas as pd
import numpy as np
import streamlit as st
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# =============================================================================
# Part 1: Model Training and Preprocessing (Typically run once)
# This section preprocesses the data, trains the model, and saves the necessary
# artifacts (model, scaler, column list) using pickle. For deployment, you would
# run this section once to generate the .pkl files and then comment it out.
# =============================================================================

def train_and_pickle_model():
    """
    Loads data, preprocesses it, trains a logistic regression model,
    and pickles the model, scaler, and feature columns.
    """
    # Load the dataset
    df = pd.read_csv('bank_with_missing.csv')

    # --- Preprocessing Steps from the Notebook ---

    # 1. Handle missing numerical data with the mean
    mean_age = df['age'].mean()
    df['age'].fillna(mean_age, inplace=True)

    # 2. Handle missing categorical data with the mode
    mode_job = df['job'].mode()[0]
    df['job'].fillna(mode_job, inplace=True)

    # 3. Drop rows with missing 'balance' values (as per notebook)
    df.dropna(subset=['balance'], inplace=True)

    # 4. Encode the target variable 'y'
    le = LabelEncoder()
    df['y'] = le.fit_transform(df['y'])

    # 5. One-Hot Encode categorical features
    x_raw = df.drop(columns=['y'])
    x = pd.get_dummies(x_raw, drop_first=True).astype(int)

    # Convert float columns in x to int where appropriate
    for col in ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']:
        if col in x.columns:
            x[col] = x[col].astype(int)

    y = df['y']

    # --- Train the Model and Scaler ---

    # Splitting the data (optional for final deployment model, but good practice)
    # For the final model, we could train on all data. Here we follow the notebook's split.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

    # Fit the scaler on the training data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    # Train the logistic regression model
    classifier = LogisticRegression(max_iter=1000, random_state=42) # Increased max_iter for convergence
    classifier.fit(x_train_scaled, y_train)

    # --- Save Artifacts to Files ---
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(classifier, model_file)

    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    with open('columns.pkl', 'wb') as columns_file:
        pickle.dump(x.columns, columns_file)

    print("Model, scaler, and columns have been trained and saved to .pkl files.")

# To create the pickle files, uncomment and run the following line once:
# train_and_pickle_model()

# =============================================================================
# Part 2: Streamlit Web Application
# This section loads the pickled artifacts and creates a web interface for users
# to get predictions from the trained model.
# =============================================================================

# --- Load the Trained Artifacts ---
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('columns.pkl', 'rb') as columns_file:
        feature_columns = pickle.load(columns_file)
except FileNotFoundError:
    st.error("Model files not found. Please run the training function first.")
    st.stop()


# --- Main Application Interface ---
st.set_page_config(page_title="Bank Term Deposit Prediction", layout="wide")
st.title('🏦 Bank Term Deposit Subscription Prediction')
st.markdown("Enter customer details to predict if they will subscribe to a term deposit.")

# --- Create Input Fields for User Data ---
st.sidebar.header("Customer Information")

# Use two columns for better layout
col1, col2 = st.sidebar.columns(2)

with col1:
    age = st.number_input('Age', min_value=18, max_value=100, value=40)
    job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
    marital = st.selectbox('Marital Status', ['married', 'single', 'divorced'])
    education = st.selectbox('Education', ['primary', 'secondary', 'tertiary', 'unknown'])
    default = st.selectbox('Has Credit in Default?', ['no', 'yes'])
    balance = st.number_input('Average Yearly Balance (€)', value=1500)
    housing = st.selectbox('Has Housing Loan?', ['no', 'yes'])
    loan = st.selectbox('Has Personal Loan?', ['no', 'yes'])

with col2:
    contact = st.selectbox('Contact Communication Type', ['cellular', 'telephone', 'unknown'])
    day = st.slider('Last Contact Day of Month', 1, 31, 15)
    month = st.selectbox('Last Contact Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    duration = st.number_input('Last Contact Duration (seconds)', value=250, min_value=0)
    campaign = st.number_input('Number of Contacts in this Campaign', value=1, min_value=1)
    pdays = st.number_input('Days since last contacted from previous campaign (-1 for new client)', value=-1)
    previous = st.number_input('Number of Contacts before this Campaign', value=0, min_value=0)
    poutcome = st.selectbox('Outcome of Previous Campaign', ['failure', 'other', 'success', 'unknown'])


# --- Prediction Logic ---
if st.sidebar.button('Predict Subscription', use_container_width=True):
    # 1. Create a dictionary from the inputs
    input_data = {
        'age': age, 'job': job, 'marital': marital, 'education': education, 'default': default,
        'balance': balance, 'housing': housing, 'loan': loan, 'contact': contact, 'day': day,
        'month': month, 'duration': duration, 'campaign': campaign, 'pdays': pdays,
        'previous': previous, 'poutcome': poutcome
    }

    # 2. Convert the dictionary into a pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # 3. One-hot encode the categorical variables
    input_df_encoded = pd.get_dummies(input_df).astype(int)

    # 4. Align the columns with the training data columns
    # This is crucial: ensures the input has the exact same features as the model was trained on
    input_df_aligned = input_df_encoded.reindex(columns=feature_columns, fill_value=0)

    # 5. Scale the data using the loaded scaler
    input_scaled = scaler.transform(input_df_aligned)

    # 6. Make a prediction and get the probability
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # --- Display the Result ---
    st.subheader('Prediction Result')
    if prediction[0] == 1:
        st.success('✅ The client is **LIKELY** to subscribe to a term deposit.')
        st.write(f"Confidence: **{prediction_proba[0][1]*100:.2f}%**")
    else:
        st.error('❌ The client is **UNLIKELY** to subscribe to a term deposit.')
        st.write(f"Confidence of non-subscription: **{prediction_proba[0][0]*100:.2f}%**")

    # Display the processed data for transparency
    with st.expander("Show Processed Input Data"):
        st.write("This is the one-hot encoded and aligned data sent to the model for prediction:")
        st.dataframe(input_df_aligned)
