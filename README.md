# Coursera: Introduction to Neural Networks and Pytorch: Final Project

This project focuses on building a logistic regression model in PyTorch to predict League of Legends match outcomes using in-game statistics. It covers data preprocessing, model training, and optimization with techniques like L2 regularization and hyperparameter tuning. Model performance is evaluated using metrics such as accuracy and ROC-AUC, while visualizations like confusion matrices and ROC curves provide insights. The project also includes feature importance analysis and demonstrates saving and loading trained models for practical applications.

**Project Overview:**

**Table of Contents**
- Introduction
  - Objectives
- Detailed Overview
  - Step 1: Data Loading and Preprocessing
  - Step 2: Logistic Regression Model
  - Step 3: Model Training
  - Step 4: Model Optimization and Evaluation
  - Step 5: Visualization and Interpretation
  - Step 6: Model Saving and Loading
  - Step 7: Hyperparameter Tuning
  - Step 8: Feature Importance

**Introduction:**

In this final project, you will embark on an exciting journey to build a logistic regression model aimed at predicting the outcomes of League of Legends matches. Leveraging various in-game statistics, this project will utilize your knowledge of PyTorch, logistic regression, and data handling to create a robust predictive model. League of Legends, a popular multiplayer online battle arena (MOBA) game, generates extensive data from matches, providing an excellent opportunity to apply machine learning techniques to real-world scenarios.

**Objectives:**

Load and preprocess the dataset: Understand and prepare the data for model training.
Implement and train a logistic regression model: Develop a model to predict match outcomes.
Evaluate model performance using appropriate metrics: Use various metrics to assess model accuracy and reliability.
Optimize the model using gradient descent and other techniques: Enhance model performance through optimization.
Interpret and visualize the results: Gain insights from the model's predictions through visualization.
Save and load the trained model: Learn techniques to persist and reload models.
Perform hyperparameter tuning: Fine-tune the model for optimal performance.

**Detailed Overview:**

Step 1: Data Loading and Preprocessing
Task 1: Load the League of Legends dataset and preprocess it for training.

This step involves reading the data, splitting it into training and testing sets, and standardizing the features. You will prepare the data for model training utilizing pandas for data manipulation, train_test_split from sklearn for data splitting, and StandardScaler for feature scaling. The data is then converted into PyTorch tensors, which are essential for PyTorch-based model training.

Step 2: Logistic Regression Model
Task 2: Implement a logistic regression model using PyTorch.

Here, you will define the logistic regression model by specifying the input dimensions and the forward pass using the sigmoid activation function. You will set the stage for training the logistic regression model by initializing the model, loss function, and optimizer.

Step 3: Model Training
Task 3: Train the logistic regression model on the dataset.

The training loop will run for a specified number of epochs. In each epoch, the model makes predictions, calculates the loss, performs backpropagation, and updates the model parameters. This iterative process helps in optimizing the model to accurately predict match outcomes.

Step 4: Model Optimization and Evaluation
Task 4: Implement optimization techniques and evaluate the model's performance.

Optimization techniques such as L2 regularization (Ridge Regression) help in preventing overfitting. The model is retrained with these optimizations, and its performance is evaluated on both training and testing sets to ensure robustness.

Step 5: Visualization and Interpretation
Task 5: Visualize the model's performance and interpret the results.

Visualization tools like confusion matrices and ROC curves provide insights into the model's performance. The confusion matrix helps in understanding the classification accuracy, while the ROC curve illustrates the trade-off between sensitivity and specificity.

Step 6: Model Saving and Loading
Task 6: Save and load the trained model.

This task demonstrates the techniques to persist a trained model using torch.save and reload it using torch.load. Evaluating the loaded model ensures that it retains its performance, making it practical for deployment in real-world applications.

Step 7: Hyperparameter Tuning
Task 7: Perform hyperparameter tuning to find the best learning rate.

By testing different learning rates, you will identify the optimal rate that provides the best test accuracy. This fine-tuning is crucial for enhancing model performance.

Step 8: Feature Importance
Task 8: Evaluate feature importance to understand the impact of each feature on the prediction.

Understanding feature importance helps in identifying which in-game statistics are most influential in predicting match outcomes. This step involves extracting the weights of the linear layer and visualizing them.

The **grading criteria** overview is shown to the learners so they know what they will be graded on and the number of points for each criteria.

There are eight tasks,  for each task, you should take a screenshot of the output of each cell with the code that generated it, as directed in the final project Jupyter Notebook.  You should provide the code used to obtain the answer in your screenshot. The provided code must run without error. Questions are independent. If you cannot answer a question or the answer you submitted is incorrect, you can still do the rest of the tasks. 

Task 1: Load the League of Legends dataset and preprocess it for training (10 points)

Import necessary libraries (pandas, train_test_split, StandardScaler, torch).

Load the dataset using pd.read_csv.

Split data into features (X) and target (y).

Use train_test_split to split the dataset.

Standardize the features using StandardScaler.

Convert data to PyTorch tensors.

Task 2: Implement a logistic regression model using PyTorch ( 5 points)

Import torch.nn and torch.optim.

Define a class LogisticRegressionModel inheriting from nn.Module.

Implement __init__ and forward methods.

Initialize the model, loss function (nn.BCELoss), and optimizer (optim.SGD).

Task 3: Train the logistic regression model on the dataset. (5 Points)

Implement the train loop for a specified number of epochs.

Make predictions and calculate the loss.

Perform backpropagation and update the model parameters.

Evaluate the model and print accuracy on training and testing sets.

Task 4: Implement optimization techniques and evaluate the model's performance. (5 points)

Implement L2 regularization in the optimizer (weight_decay parameter).

Retrain the model with the same training loop.

Evaluate the optimized model on training and testing sets.

Task 5: Visualize the model's performance and interpret the results. (8 Points)

Import necessary libraries (matplotlib.pyplot, confusion_matrix, classification_report, roc_curve, auc).

Generate and plot the confusion matrix.

Print the classification report.

Plot the ROC curve and calculate the AUC.

Task 6: Save and load the trained model. (8 Points)

Use torch.save to save the model's state dictionary.

Use torch.load to load the state dictionary into a new model instance.

Set the loaded model to evaluation mode.

Evaluate the loaded model and ensure consistent performance.

Task 7: Perform hyperparameter tuning to find the best learning rate. (8 Points)

Define a list of learning rates to test.

Reinitialize the model and optimizer for each learning rate.

Train and evaluate the model for each learning rate.

Print the best learning rate and corresponding test accuracy.

Task 8: Evaluate feature importance to understand the impact of each feature on the prediction. (6 Points)

Extract the weights from the linear layer.

Create a DataFrame with feature names and their corresponding importance.

Sort the DataFrame by importance.

Plot the feature importance using a bar plot.
