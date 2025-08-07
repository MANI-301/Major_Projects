# Future Sales Prediction
The aim is to build a predictive model and find out the sales of each product at a particular store. Using this model, Big Marts like Target will try to understand the properties of products and stores which play a key role in increasing sales. So the idea is to find out the properties of a product, and store which impacts the sales of a product. We came up with certain hypothesis in order to solve the problem statement relating to the various factors of the products and stores. We develop a predictive model using data science for forecasting the sales of a business such as Target Corporation
Future Sales Prediction

This README provides instructions for executing the Python code for sales prediction with data from the dataset "Sales.csv."

Before running the code, make sure you have the necessary Python libraries installed. You can install them using `pip` in the command prompt

- NumPy: import numpy as np
- Pandas: import pandas as pd
- Scikit-Learn: from sklearn.metrics import mean_squared_error, r2_score
- Matplotlib: import matplotlib.pyplot as plt
- Seaborn: import seaborn as sns
- Plotly Express: import plotly.express as px
- Scikit-Learn for model selection: from sklearn.model_selection import train_test_split
- Scikit-Learn for linear regression: from sklearn.linear_model import LinearRegression
- Scikit-Learn for data preprocessing: from sklearn.preprocessing import StandardScaler

Data Preparation

1. Import the dataset
   Import the "Sales.csv" data set using pandas library
   
2. Display the data
   
3. Get the shape of the dataset (rows, columns)
   
4. Get information about the dataset
  
5. Get a description of the dataset

6. Check for missing values
   
   
Data Visualization

Visualize the data using Plotly Express
   - Plot representation of TV data
   - Plot representation of Newspaper data
   - Plot representation of Radio data
   - Heatmap representation of correlation
   - Histogram of data
   
   

Data Preprocessing

1. Split the data into features (x) and the target (y)
   
2. Split the data into training and testing sets
   
3. Create a StandardScaler instance
   
4. Fit the scaler on the training data and transform it
    
5. Transform the test data using the same scaler
    

Model Building and Evaluation

1. Create a Linear Regression model
    
2. Fit the model on the scaled training data
    
3. Make predictions on the scaled test data
    
4. Evaluate the model's performance
    - Calculate Mean Squared Error (MSE)
    - Calculate R-squared Score
    - Print the evaluation results
    
5. Display the results in a DataFrame
    

Visualization of Results

1. Create a line graph to compare actual and predicted sales
    

Conclusion

This code provides a step-by-step guide for data preparation, data visualization, data preprocessing,
model building, evaluation, and result visualization for future sales prediction.

This code provides a step-by-step guide for data preparation, data visualization, data preprocessing, model building, evaluation, and result visualization for future sales prediction.
