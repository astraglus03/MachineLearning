import pandas as pd
import numpy as np

df=pd.read_csv('http://bit.ly/perch_csv')
perch_full=df.to_numpy()

# print(perch_full)

from sklearn.preprocessing import PolynomialFeatures
#

#degree=2

poly=PolynomialFeatures()
poly.fit([[2,3]])

# 1(bias), 2, 3, 2**2,2*3, 3**2
print(poly.transform([[2,3]]))


perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

from sklearn.model_selection import train_test_split

train_input,test_input,train_target,test_target = train_test_split(perch_full,perch_weight, random_state=42)

poly=PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly=poly.transform(train_input)
print(train_poly.shape)

poly.get_feature_names_out()

test_poly=poly.transform(test_input)


from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(train_poly,train_target)

print(lr.score(train_poly,train_target))
print(lr.score(test_poly,test_target))

poly=PolynomialFeatures(degree=5, include_bias=False) # 5차항까지 추가한다면?

poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

print(train_poly.shape)

lr.fit(train_poly,train_target)

print(lr.score(train_poly,train_target))
print(lr.score(test_poly,test_target))

# -------------Lasso Ridge--------------------
# 다른데이터가 들어오더라도 맞춰주는 역할(과대적합을 방지하기위해 사용)

from sklearn.preprocessing import StandardScaler

ss=StandardScaler() # 평균 표준편차 계산과정을 이 함수가 해줌
ss.fit(train_poly)

train_scaled=ss.transform(train_poly)
test_scaled=ss.transform(test_poly)

from sklearn.linear_model import Ridge

ridge=Ridge()
ridge.fit(train_scaled,train_target)

print(ridge.score(train_scaled,train_target))
print(ridge.score(test_scaled,test_target))

train_score=[]
test_score=[]

alpha_list=[0.001,0.01,0.1,1,10,100]

for i in alpha_list:
    ridge = Ridge(i=i)
    ridge.fit(train_scaled,train_target)
    train_score.append(ridge.score(train_scaled,train_target))
    test_score.append(ridge.score(test_scaled,test_target))
