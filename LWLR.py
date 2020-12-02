import requests, zipfile, io, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load elo rankings
with open("eloVals.json") as file:
    elos = json.load(file)


# Load csv
data_orig = pd.read_csv("allSeasonScores.csv")

'''
jsonObject
jsonObj[teamAbbr][year][week][unit]
where unit is either 'off' or 'def'
'''

#Comment