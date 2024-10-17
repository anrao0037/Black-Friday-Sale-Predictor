# Black Friday Sale Predictor

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## Overview
The **Black Friday Sale Predictor** is a machine learning project aimed at predicting the purchase amount for customers during Black Friday sales. The model analyzes customer demographics and product information to provide insights into purchase patterns, helping businesses maximize their sales strategies. The dataset consists of customer data, including gender, age, location, and historical purchase amounts.

## Features
- **Predict Purchase Amount**: Predicts how much a customer is likely to spend based on input features such as demographics and product categories.
- **Data Preprocessing**: Cleans and processes data to handle missing values, categorical variables, and feature scaling.
- **Multiple Algorithms**: Implements multiple machine learning models (e.g., Linear Regression, Decision Trees, Random Forest, XGBoost) for predicting sales.
- **Model Evaluation**: Compares models based on performance metrics like RMSE, MAE, and R² score.

## Dataset
The dataset used for this project is sourced from a Black Friday sales dataset, containing over 500,000 rows of customer and purchase details. Key features include:
- **User_ID**: Unique identifier for each customer.
- **Product_ID**: Unique identifier for each product.
- **Gender**: Gender of the customer.
- **Age**: Age group of the customer.
- **Occupation**: Occupation code for the customer.
- **City_Category**: Category of the city where the customer resides.
- **Stay_In_Current_City_Years**: Number of years the customer has stayed in the current city.
- **Marital_Status**: Marital status of the customer.
- **Product_Category_1, Product_Category_2, Product_Category_3**: Product category codes.

The dataset can be found [here](https://www.kaggle.com/sdolezel/black-friday).

## Technologies Used
- **Programming Language**: Python
- **Libraries**: 
  - Pandas (for data manipulation)
  - NumPy (for numerical operations)
  - Scikit-learn (for machine learning models)
  - XGBoost (for gradient boosting model)
  - Matplotlib, Seaborn (for data visualization)
- **IDE**: Jupyter Notebook, PyCharm, or any Python-compatible IDE

## Setup Instructions

### Prerequisites
- Python 3.x
- Required Python libraries (listed in `requirements.txt`)

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/anrao0037/Black-Friday-Sale-Predictor.git
   ```

2. **Install Required Libraries**:
   Navigate to the project directory and install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   Download the dataset from [Kaggle](https://www.kaggle.com/sdolezel/black-friday) and place it in the `data/` folder of the project directory.

## Usage
1. **Data Preprocessing**:
   - Run the `data_preprocessing.py` script to clean and preprocess the data. This will handle missing values and perform necessary encoding for categorical features.

2. **Training the Model**:
   - Train the machine learning models by running `train_model.py`. This script will train various models and evaluate them based on the dataset.
   ```bash
   python train_model.py
   ```

3. **Prediction**:
   - Once the model is trained, you can use the `predict.py` script to make predictions on new customer data. Update the customer information in the script and run the command:
   ```bash
   python predict.py
   ```

4. **Evaluation**:
   - The trained models will be evaluated using various metrics, and results will be displayed for comparison. The best-performing model will be saved as a `.pkl` file for future predictions.

## Model Performance
The models are evaluated based on:
- **Root Mean Squared Error (RMSE)**: Measures the difference between predicted and actual values.
- **Mean Absolute Error (MAE)**: Provides an average of the absolute differences between predicted and actual values.
- **R² Score**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

Final model performance can vary depending on the dataset and features used.

## Contributing
Contributions are welcome! If you would like to contribute, please fork the repository and submit a pull request. Ensure that your changes align with the project's goals and coding standards.

