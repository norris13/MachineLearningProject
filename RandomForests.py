
#Use pandas for messing with the csv
import pandas as pd
#Use matplotlib for plots
import matplotlib.pyplot as plt
# Use numpy to convert to arrays
import numpy as np
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

data_orig = pd.read_csv("teamElos.csv")
data = data_orig.copy()

print(data.columns)
features = data.drop('Offensive Elo Rating', axis=1)
labels = data['Offensive Elo Rating']

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)