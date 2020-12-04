import json
import math
#Use pandas for messing with the csv
import pandas as pd
#Use matplotlib for plots
import matplotlib.pyplot as plt
# Use numpy to convert to arrays
import numpy as np
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import metrics
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

#jsonObj[teamAbbr][year][week][unit]
#where unit is either 'off' or 'def'

with open("eloVals.json") as file:
    elos = json.load(file)

data_orig = pd.read_csv("./data/allSeasonScores.csv")
data = data_orig.copy()

#print(data)
home_off_elo = []
home_def_elo = []
away_off_elo = []
away_def_elo = []

      
# Select column contents by column 
# name using [] operator 
homeTeamNames = data[data.columns[3]]
# print('Column Name : ', data.columns[3]) 
# print('Column Contents : ', homeTeamNames.values)

awayTeamNames = data[data.columns[4]]

year = data[data.columns[1]]

week = data[data.columns[2]]

#choose either off or def for unit

num_rows = data.shape[0]

#Create lists of elos so they can be appended to the dataFrame
for i in range(num_rows):
    home_off_elo.insert(i, elos[ homeTeamNames[i] ][ str(year[i]) ][ str(week[i]) ]['off'])
    home_def_elo.insert(i, elos[ homeTeamNames[i] ][ str(year[i]) ][ str(week[i]) ]['def'])
    away_off_elo.insert(i, elos[ awayTeamNames[i] ][ str(year[i]) ][ str(week[i]) ]['off'])
    away_def_elo.insert(i, elos[ awayTeamNames[i] ][ str(year[i]) ][ str(week[i]) ]['def'])

elo_data = data.assign(homeOffElo = home_off_elo, awayOffElo = away_off_elo, homeDefElo = home_def_elo, awayDefElo = away_def_elo)
#print(elo_data)

features = elo_data.drop( ['homeOffPoints', 'count', 'homeTeam', 'awayTeam', 'week', 'season',], axis=1 )

target = elo_data['homeOffPoints']

feature_list = list(features.columns)

#Convert features and labels into numpy arrays
features  = np.array(features)
labels = np.array(target)


# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

# print('Training Features Shape:', train_features.shape)
# print('Training Labels Shape:', train_labels.shape)
# print('Testing Features Shape:', test_features.shape)
# print('Testing Labels Shape:', test_labels.shape)

# print(train_features)
# print(train_labels)

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

print(predictions[:20])
print(test_labels[:20])

print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))

#mse = mean_squared_error(test_labels, predictions)
#rmse = math.sqrt(mse)
#print('Accuracy for Random Forest',100*max(0,rmse)) 
#print('Accuracy for Random Forest',max(0,mse)) 