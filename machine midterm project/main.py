import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

moqipath = pd.read_excel("./mosquito/export.xlsx")
bopath = pd.read_excel("./mac/boryung/export.xlsx")
bupath = pd.read_excel("./mac/buyeo/export.xlsx")
chpath = pd.read_excel("./mac/cheonan/export.xlsx")
gepath = pd.read_excel("./mac/geumsan/export.xlsx")
sepath = pd.read_excel("./mac/seosan/export.xlsx")
rainpath= pd.read_excel("./exportrain.xlsx")

temp_df = pd.concat([bopath,bupath,chpath,gepath,sepath],axis=1)

temp_df['meantemp'] = temp_df["평균기온(°C)"].mean(axis=1)
temp_df['meanhu'] = temp_df["평균 상대습도(%)"].mean(axis=1)
temp_df.to_excel('./export.xlsx', sheet_name='new_name')
moqipath.rename(columns={0:"moqicount"},inplace=True)
total_df = pd.concat([temp_df["meantemp"], temp_df["meanhu"],moqipath["moqicount"],rainpath["meanrain"]], axis=1)
total_df.to_excel('./exportfinal.xlsx', sheet_name='new_name')
train = pd.read_excel('exportfinal.xlsx')

# print("데이터")
# print(train.head())

def up(value):
  return round(value,1)
def tr(value):
    return round(value,0)

train['meantemp'] = train.meantemp.apply(up)
train['meanhu'] = train.meanhu.apply(up)
train['meanrain'] = train.meanrain.apply(up)
train['moqicount'] = train.moqicount.apply(tr)

inputdata = train[['meanhu', 'meantemp','meanrain']]
data_input = train[['meanhu', 'meantemp','meanrain']].to_numpy()
target_input = train['moqicount'].to_numpy()

# print(data_input)

#훈련 세트와 테스트 세트

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data_input, target_input,random_state=7)

# print(np.shape(train_input))
# print(np.shape(train_target))

#표준화 전처리
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

#K-최근접 이웃 분류기의 확률 예측
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print("학습 모델 확률 : ", kn.score(train_scaled, train_target))
print("테스트 모델 확률 : ", kn.score(test_scaled, test_target))


#다중 회귀
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3,include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)
test_poly = poly.transform(test_input)

#모델 훈련
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

#릿지 회귀
from sklearn.linear_model import Ridge
ridge = Ridge(0.01)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)

rf.fit(train_input, train_target)

y_pred = rf.predict(test_input)

mse = mean_squared_error(test_target, y_pred)
r2 = r2_score(test_target, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

features = inputdata.columns
importances = rf.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10,5))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


plt.plot(train["meantemp"])
plt.plot(train["meanhu"])
plt.plot(train["meanrain"])
plt.show()

plt.plot(train["moqicount"])
plt.show()