import requests, zipfile, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load csv
data_orig = pd.read_csv("allSeasonScores.csv")

'''
jsonObject
jsonObj[teamAbbr][year][week][unit]
where unit is either 'off' or 'def'
'''