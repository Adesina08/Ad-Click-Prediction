# Ad Click Prediction App 🖱️ 🎯

This project is a **Streamlit-based web app** that predicts whether a user will click on an advertisement based on various user features like age, gender, device type, and browsing history. The app allows users to explore the dataset, train different machine learning models, and make predictions using the trained model.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [How to Use the App](#how-to-use-the-app)
- [Contributing](#contributing)

## Features
- **Data Exploration**: Explore the ad click dataset, view basic statistics, and visualize correlations.
- **Model Training**: Train three different machine learning models (Random Forest, Logistic Regression, XGBoost) and evaluate their accuracy.
- **Make Predictions**: Use the trained model to predict whether a user will click on an advertisement based on input variables like age, gender, device type, and more.
- **Feature Importance**: Visualize the importance of each feature in the selected model.

## Technologies Used
- **Python 3.11**
- **Streamlit**
- **pandas**
- **scikit-learn**
- **xgboost**
- **matplotlib**
- **seaborn**

## Setup and Installation

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/ad-click-prediction-app.git
cd ad-click-prediction-app 
```
### 2. Install the dependencies:
```bash
pip install -r requirements.txt
```
### 3. Run the Streamlit app:
```bash
streamlit run app.py
```

## How to Use the App
1. **Data Exploration:** Start by exploring the dataset. View a preview of the data and visualize the correlation between features.
2. **Model Training:** Select a machine learning model (Random Forest, Logistic Regression, or XGBoost) from the sidebar and train it using the dataset. The app will display the model's accuracy and a classification report.
3. **Make Predictions:** Use the last trained model to make predictions based on user-provided inputs. You can input values for age, gender, device type, and more to get a prediction on whether the user will click an ad.


## Contributing
If you would like to contribute to this project, feel free to fork the repository and submit a pull request. Contributions are welcome!
