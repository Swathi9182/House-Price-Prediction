# 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview

This project predicts house prices using Machine Learning.
A real-world housing dataset is used to train and evaluate
multiple regression algorithms.

The project also includes an interactive Streamlit web application
where users can enter house details and receive an estimated house price.

## 🎯 Objectives

- Understand and preprocess housing data
- Select features and target variables
- Train multiple Machine Learning models
- Compare model performance
- Evaluate predictions using regression metrics
- Build an interactive prediction application

## 📊 Dataset

The project uses the California Housing dataset.

### Features

- MedInc
- HouseAge
- AveRooms
- AveBedrms
- Population
- AveOccup
- Latitude
- Longitude

### Target

- MedHouseVal

## 🤖 Machine Learning Algorithms

The following regression algorithms were implemented:

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression
4. Support Vector Regression (SVR)

## 📏 Evaluation Metrics

The models were evaluated using:

- MAE — Mean Absolute Error
- MSE — Mean Squared Error
- RMSE — Root Mean Squared Error
- R² Score

## 📈 Model Comparison

The models were compared using the same test dataset.

Random Forest achieved an R² score of approximately 0.805
in our experiment and produced the lowest error among the
three models initially compared.

## 🖥️ Application

The project includes a Streamlit web application.

Users can enter:

- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Average Occupancy
- Latitude
- Longitude

The application then displays an estimated house price.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

## 📁 Project Structure

```text
House_Price_Prediction/
│
├── main.py
├── app.py
├── house_price_model.pkl
├── requirements.txt
└── README.md