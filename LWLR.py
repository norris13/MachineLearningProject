import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn import metrics

'''
jsonObject
jsonObj[teamAbbr][year][week][unit]
where unit is either 'off' or 'def'
'''

# Function definitions
def parseName(name):
    switcher = {
        "HO" : "homeOffPoints",
        "AO" : "awayOffPoints",
        "HD" : "homeDefPoints",
        "AD" : "awayDefPoints"
    }
    return switcher.get(name, "nothing")

def dataPick():
    name = input("Enter what data you want to see (HO for home offense, AO for away offense, HD for home defense, AD for away defense): ")
    parsed = parseName(name)
    if parsed == "nothing":
        print("Please try again.")
        return dataPick()
    return parsed

# Ask what data the user wants
dataName = dataPick()
print(dataName)

# Load elo rankings
with open("eloVals.json") as file:
    elos = json.load(file)

# Load csv
data_orig = pd.read_csv("allSeasonScores.csv")
data = data_orig.copy()

# Create arrays for elos
homeOff = []
homeDef = []
awayOff = []
awayDef = []

# Parse data for what we need
homeTeamNames = data[data.columns[3]]
awayTeamNames = data[data.columns[4]]
year = data[data.columns[1]]
week = data[data.columns[2]]

rows = data.shape[0]

# Append elos to dataFrame and format dataFrame
for i in range(rows):
    homeOff.insert(i, elos[ homeTeamNames[i] ][ str(year[i]) ][ str(week[i]) ]['off'])
    homeDef.insert(i, elos[ homeTeamNames[i] ][ str(year[i]) ][ str(week[i]) ]['def'])
    awayOff.insert(i, elos[ awayTeamNames[i] ][ str(year[i]) ][ str(week[i]) ]['off'])
    awayDef.insert(i, elos[ awayTeamNames[i] ][ str(year[i]) ][ str(week[i]) ]['def'])

elo_data = data.assign(homeOffElo = homeOff, awayOffElo = awayOff, homeDefElo = homeDef, awayDefElo = awayDef)
# print(elo_data)

# Format data for conversion
features = elo_data.drop( [dataName, 'homeTeam', 'awayTeam'], axis=1 )
target = elo_data[dataName]

# Convert features and labels into numpy arrays
features  = np.array(features)
labels = np.array(target)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 13)

'''
# Flatten arrays for isotonic regression
# trainFeat1d = train_features.flatten() 
# trainLab1d = train_labels.flatten()
# testFeat1d = test_features.flatten()
# testLab1d = test_labels.flatten()
'''

# Linear Regression
lr = LinearRegression()
fitted = lr.fit(train_features, train_labels)
lrPred = lr.predict(test_features)

# Output Error
print("------------------------------------------------------------------------------------------------------------------------")
print('LR Mean Absolute Error:', metrics.mean_absolute_error(test_labels, lrPred))
print('LR Mean Squared Error:', metrics.mean_squared_error(test_labels, lrPred))
print('LR Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, lrPred)))

# Plot
fig = plt.figure()
plt.plot(train_features, train_labels, 'r.', markersize = 2)
plt.plot(test_features, lrPred, 'b-', markersize = 2)
plt.legend(('Data', 'Linear Fit'), loc='lower right')
plt.show()

'''
# Isotonic Regression
ir = IsotonicRegression()
ir.fit(train_features[0], train_labels[0])
irPred = ir.predict(test_features[0])

print('IR Mean Absolute Error:', metrics.mean_absolute_error(test_labels[0], irPred))
print('IR Mean Squared Error:', metrics.mean_squared_error(test_labels[0], irPred))
print('IR Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels[0], irPred)))
'''
