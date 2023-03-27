import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fish=pd.read_csv('https://bit.ly/fish_csv_data')
fish.head() # 처음 행 5개를 출력하게 해줌

print(pd.unique(fish['Species'])) # species열에서 고유한값 추출할때 unique사용


fish_input=fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()

# print(fish_input[:5])

fish_target=fish['Species'].to_numpy()
# 열 선택할떄 괄호 두개 사용하면 fish_target이 2차원 배열이 되어버림

from sklearn.model_selection import train_test_split

train_input,test_input,train_target,test_target=train_test_split(fish_input,fish_target, random_state=42)

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()
ss.fit(train_input)
train_scaled=ss.transform(train_input)
test_scaled=ss.transform(test_input)

from sklearn.neighbors import KNeighborsClassifier

kn=KNeighborsClassifier()
kn.n_neighbors=3
kn.fit(train_scaled,train_target)
print(kn.score(train_scaled,train_target))
print(kn.score(test_scaled,test_target))

print(kn.classes_)
print(kn.predict(test_scaled[:5]))
print("-------------------------")

proba=kn.predict_proba(test_scaled[:5]) # predict_proba는 classes_ 속성과 동일함
# print(proba)
print(np.round(proba,decimals=3)) # decimal은 소수점 n번째자리까지 표기.

distances,indexes = kn.kneighbors(test_scaled[3:4]) # kneighbors는 2차원 배열이여야하고 슬라이싱 사용하면 항상 2차원 배열
print(train_target[indexes])

bream_smelt_indexes=(train_target=='Bream') | (train_target=='Smelt')
train_bream_smelt=train_scaled[bream_smelt_indexes]
target_bream_smelt=train_target[bream_smelt_indexes]

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(train_bream_smelt,target_bream_smelt)

# print(lr.predict(train_bream_smelt[:5]))
print(lr.predict_proba(train_bream_smelt[:5])) # 예측 확률 = predict_proba

print(lr.classes_) # 타겟값 알파벳순 정렬
print(lr.coef_,lr.intercept_)

decisions=lr.decision_function(train_bream_smelt[:5])
print(decisions)

from scipy.special import expit
print(expit(decisions))

lr=LogisticRegression(C=20, max_iter=1000) # C는 규제, max_iter는 반복적인 계산 횟수
lr.fit(train_scaled,train_target)
print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target))

proba=lr.predict_proba(test_scaled[:5])
# print(np.round(proba,decimals=3))

print(lr.coef_.shape,lr.intercept_.shape)

decision=lr.decision_function(test_scaled[:5])
print(np.round(decision,decimals=2))

from scipy.special import softmax

proba=softmax(decision,axis=1)
print(np.round(proba,decimals=3))