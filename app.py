import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings("ignore")

# Path to the dataset
dataset_path = "ad_click_dataset.csv"


# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv(dataset_path)
    return data


# Preprocess the dataset
def preprocess_data(df):
    # Handle missing values
    df["age"].fillna(df["age"].median(), inplace=True)
    df["gender"].fillna("Unknown", inplace=True)
    df["browsing_history"].fillna("Unknown", inplace=True)
    df["time_of_day"].fillna("Unknown", inplace=True)
    df["ad_position"].fillna("Unknown", inplace=True)

    # Encode categorical features
    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"])
    df["device_type"] = le.fit_transform(df["device_type"].fillna("Unknown"))
    df["ad_position"] = le.fit_transform(df["ad_position"])
    df["browsing_history"] = le.fit_transform(df["browsing_history"])
    df["time_of_day"] = le.fit_transform(df["time_of_day"])

    # Split features and target
    X = df.drop(columns=["id", "full_name", "click"])  # Dropping unnecessary columns
    y = df["click"]

    return X, y


# Visualize correlation heatmap
def plot_heatmap(df):
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.write("No numeric columns found for correlation heatmap.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)


def plot_feature_importance(model, X):
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=importance[sorted_idx], y=X.columns[sorted_idx], palette="viridis", ax=ax
    )
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)


# Main Streamlit app
def main():
    st.markdown(
        """
    <style>
    body {
        background-color: #f4f4f4;
        font-family: 'Arial', sans-serif;
    }
    .main-title {
        color: #4CAF50;
        font-size: 36px;
        font-weight: bold;
    }
    .sub-title {
        color: #4CAF50;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 9999;
    }
    .modal-content {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h1 class='main-title'>Ad Click Prediction App</h1>", unsafe_allow_html=True
    )

    # Sidebar for data and model selection
    st.sidebar.title("Options")
    option = st.sidebar.selectbox(
        "Choose an option", ["Data Exploration", "Model Training", "Make Predictions"]
    )

    # Load data
    df = load_data()

    # Initialize session state
    if "model" not in st.session_state:
        st.session_state.model = None
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    if "show_modal" not in st.session_state:
        st.session_state.show_modal = False
        st.session_state.result = ""

    if option == "Data Exploration":
        st.markdown(
            "<h2 class='sub-title'>Data Exploration</h2>", unsafe_allow_html=True
        )
        st.write(df.head())
        st.write("Data Overview:")
        st.write(df.describe())
        plot_heatmap(df)

    elif option == "Model Training":
        st.markdown("<h2 class='sub-title'>Model Training</h2>", unsafe_allow_html=True)
        X, y = preprocess_data(df)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model selection
        model_choice = st.sidebar.selectbox(
            "Select Model", ["Random Forest", "Logistic Regression", "XGBoost"]
        )

        if model_choice == "Random Forest":
            st.session_state.model = RandomForestClassifier()
        elif model_choice == "Logistic Regression":
            st.session_state.model = LogisticRegression()
        else:
            st.session_state.model = XGBClassifier()

        st.session_state.model.fit(X_train, y_train)
        y_pred = st.session_state.model.predict(X_test)

        st.write("Model Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        if model_choice in ["Random Forest", "XGBoost"]:
            plot_feature_importance(st.session_state.model, X)

        st.session_state.model_trained = True

    elif option == "Make Predictions":
        st.markdown(
            "<h2 class='sub-title'>Make Predictions</h2>", unsafe_allow_html=True
        )

        if not st.session_state.model_trained:
            st.warning("Please train a model first in the 'Model Training' section.")
        else:
            # User input form
            user_input = {
                "age": st.number_input("Age", min_value=18, max_value=100),
                "gender": st.selectbox(
                    "Gender", ["Male", "Female", "Non-Binary", "Unknown"]
                ),
                "device_type": st.selectbox(
                    "Device Type", ["Desktop", "Mobile", "Tablet", "Unknown"]
                ),
                "ad_position": st.selectbox("Ad Position", ["Top", "Side", "Bottom"]),
                "browsing_history": st.selectbox(
                    "Browsing History",
                    [
                        "Social Media",
                        "News",
                        "Entertainment",
                        "Education",
                        "Shopping",
                        "Unknown",
                    ],
                ),
                "time_of_day": st.selectbox(
                    "Time of Day",
                    ["Morning", "Afternoon", "Evening", "Night", "Unknown"],
                ),
            }

            # Add a "Predict" button
            if st.button("Predict"):
                # Convert input into DataFrame
                user_data = pd.DataFrame([user_input])

                # Preprocess user input
                X, _ = preprocess_data(pd.concat([df, user_data], ignore_index=True))
                user_data_processed = X.iloc[-1:]  # Get the last row (user input)

                # Prediction
                prediction = st.session_state.model.predict(user_data_processed)
                st.session_state.result = "Click" if prediction[0] == 1 else "No Click"
                st.session_state.show_modal = True

    # Modal display
    if st.session_state.show_modal:
        st.markdown(
            f"""
            <div class="modal" onclick="window.location.reload();">
                <div class="modal-content">
                    <h3>Prediction Result</h3>
                    <p style='font-size: 20px; font-weight: bold;'>{st.session_state.result}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.show_modal = False  # Reset modal state after displaying


if __name__ == "__main__":
    main()
