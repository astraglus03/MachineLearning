import pandas as pd

df=pd.read_csv('https://bit.ly/perch_csv_data')
perch_full=df.to_numpy() # 판다스로 가져와서 넘파이 규격으로 해주는게 좋음
# print(perch_full)
# print(len(perch_full))

import numpy as np

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

from sklearn.model_selection import train_test_split

train_input,test_input,train_target,test_target=train_test_split(perch_full,perch_weight,random_state=42)

from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures()

# poly.fit([[2,3]])
# print(poly.transform([[2,3]]))
#
# poly=PolynomialFeatures(include_bias=False) # include_bias=False로 지정하면 절편을 위한 항 제거
# poly.fit([[2,3]])
# print(poly.transform([[2,3]]))

poly=PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly=poly.transform(train_input)
print(train_poly.shape)

# print(poly.get_feature_names_out())
test_poly=poly.transform(test_input)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(train_poly,train_target)
print(lr.score(train_poly,train_target))

print(lr.score(test_poly,test_target))

poly=PolynomialFeatures(degree=5, include_bias=False) # 새로운 특성 생성(고차항으로 변환)
poly.fit(train_input)
train_poly=poly.transform(train_input)
test_poly=poly.transform(test_input)
print(train_poly.shape)
print(test_poly.shape)

lr.fit(train_poly,train_target)
print(lr.score(train_poly,train_target))
print(lr.score(test_poly,test_target))

from sklearn.preprocessing import StandardScaler

ss=StandardScaler() # 정규화 하기위해 사용함
ss.fit(train_poly)
train_scaled=ss.transform(train_poly)
test_scaled=ss.transform(test_poly)


from sklearn.linear_model import Ridge # 릿지와 라쏘 둘다 sklearn.linear_model에 있음 (규제 방법 두가지)

ridge=Ridge()

ridge.fit(train_scaled,train_target)
print(ridge.score(train_scaled,train_target))

print(ridge.score(test_scaled,test_target))

#alpha값이 크면 규제 강도가 세지므로 계수 값을 더 줄여 조금 더 과소적합 유도함

import matplotlib.pyplot as plt

train_score=[]
test_score=[]

alpha_list= [0.001,0.01,0.1,1,10,100]
for i in alpha_list:
    ridge=Ridge(alpha=i)
    ridge.fit(train_scaled,train_target)
    train_score.append(ridge.score(train_scaled,train_target))
    test_score.append(ridge.score(test_scaled,test_target))

plt.plot(np.log10(alpha_list),train_score) # -3이 0.001이고 나머지 5개가 나머지 배열
plt.plot(np.log10(alpha_list),test_score)
# plt.show()

# 그래프를 봤을때 -1일때 가장 점수가 높으므로 0.1을 알파로 사용하여 최종 모델 훈련

ridge=Ridge(alpha=0.1)
ridge.fit(train_scaled,train_target)
print(ridge.score(train_scaled,train_target))
print(ridge.score(test_scaled,test_target))

