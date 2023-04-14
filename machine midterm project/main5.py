import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

exportfinal = pd.read_excel("./exportfinal.xlsx")

bopath = pd.read_excel("./r_boryung.xlsx")
bupath = pd.read_excel("./r_buyeo.xlsx")
chpath = pd.read_excel("./r_seosan.xlsx")
gepath = pd.read_excel("./r_geumsan.xlsx")
sepath = pd.read_excel("./r_cheonan.xlsx")

rain_df = pd.concat([bopath,bupath,chpath,gepath,sepath],axis=1)

rain_df['meanrain'] = rain_df["강수량(mm)"].mean(axis=1)

rain_df.to_excel('./exportrain.xlsx', sheet_name='new_name')

print(rain_df['meanrain'])

