import pandas as pd
import numpy as np

ep=pd.read_excel('ep.xlsx')
fp=pd.read_excel('fp.xlsx')
gas_input=fp.to_numpy()
gas_target=ep.to_numpy()

print(len(gas_input))
print(len(gas_target))

from sklearn.model_selection import train_test_split

train_input,test_input,train_target,test_target=\
    train_test_split(gas_input,gas_target,random_state=1)
print(len(train_input))
print(len(test_input))

from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(degree=6, include_bias=False)
poly.fit(train_input)
train_poly=poly.transform(train_input)
test_poly=poly.transform(test_input)

# print(poly.get_feature_names_out())
print(train_poly.shape)
print(test_poly.shape)

from sklearn.linear_model import LinearRegression

# lr=LinearRegression()
# lr.fit(train_poly,train_target)
# print(lr.score(train_poly,train_target))
# print(lr.score(test_poly,test_target))
# print("----------------")
from sklearn.preprocessing import StandardScaler

ss=StandardScaler() # 표준 정규화 시켜주기
ss.fit(train_poly)
train_scaled=ss.transform(train_poly)
test_scaled=ss.transform(test_poly)

from sklearn.linear_model import Ridge

ridge=Ridge()
ridge.fit(train_scaled,train_target)
print(ridge.score(train_scaled,train_target))
print(ridge.score(test_scaled,test_target))

import matplotlib.pyplot as plt

train_score=[]
test_score=[]

alpha_list=[0.001,0.01,0.1,1,10,100]

for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_scaled, train_target)
    train_score.append(ridge.score(train_scaled,train_target))
    test_score.append(ridge.score(test_scaled,test_target))

plt.plot(np.log10(alpha_list),train_score)
plt.plot(np.log10(alpha_list),test_score)
plt.show()

print("----------------")
ridge=Ridge(alpha=1)
ridge.fit(train_scaled,train_target)
print(ridge.score(train_scaled,train_target))
print(ridge.score(test_scaled,test_target))