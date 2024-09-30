# Predicting Student Scores using Supervised Machine Learning
The project aim is  predicting student scores based on the number of study hours using supervised machine learning techniques.

Project Overview:
Objective: The main objective of this project is to predict the scores of students based on the number of hours they study per day.
Dataset: The dataset used for this project contains two columns - 'Hours' and 'Scores'. It consists of records indicating the number of study hours and the corresponding scores achieved by students.
Approach: Linear Regression, a supervised machine learning algorithm, is employed to establish a relationship between the number of study hours and the scores obtained.
Project Workflow:
Data Importing and Exploration:

The necessary libraries such as Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn are imported.
The dataset is loaded from a CSV file hosted on GitHub.
Initial exploration of the dataset is performed by displaying the first few rows and visualizing the data using a scatter plot to observe the relationship between study hours and scores.
Data Preparation:

The dataset is divided into input features (X) and target variable (y).
Data splitting is carried out into training and testing sets to evaluate the performance of the model.
Model Training:

A Linear Regression model is instantiated and trained using the training data.
Model Evaluation:

The trained model is evaluated using the testing data.
Evaluation metrics such as R-squared (R²) and Root Mean Squared Error (RMSE) are calculated to assess the model's performance.
Prediction:

Using the trained model, predictions are produced using the training data.
Another prediction is performed for a student studying 9.25 hours/day.
Files:
predict_student_scores.ipynb: Jupyter Notebook containing the Python code for loading and preprocessing of data, training the model, performing evaluation on that model, and generating predictions.
README.md: This is the README file containing an overview of the project including its objectives, how to work with it, and files included in the project.
student_scores.csv: This is a CSV file with data for training and testing the model.
Dependencies:
Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn
