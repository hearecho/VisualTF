import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

df = pd.read_csv('../data/weather-trends.csv',sep=';')

plt.figure()
df['temperature'].plot()
df['humidity'].plot()
plt.show()