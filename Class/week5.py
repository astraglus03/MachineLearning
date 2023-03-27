import pandas as pd # 대량 데이터 처리 할때 주로 사용
import numpy as np
from sklearn.model_selection import train_test_split # 학습용 실험용
from sklearn.preprocessing import StandardScaler # 표준
from sklearn.neighbors import KNeighborsClassifier # 주변 샘플에 따른 평균
fish = pd.read_csv('https://bit.ly/fish_csv')  # 데이터 불러오기
fish.head()
print(pd.unique(fish['Species']))
fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
# 5개 정보 가져오기(판다스로) numpy규격으로 바꾸는게 편함
print(fish_input[:5])
fish_target = fish['Species'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
ss = StandardScaler() #표준 점수화
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

from sklearn.neighbors import KNeighborsClassifier

kn= KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled,train_target)
print(kn.classes_)

print(kn.predict(test_scaled[:5]))
proba=kn.predict_proba(test_scaled[:5])
print(np.round(proba,decimals=4))


print("-------------------------------")

bream_smelt_indexes=(train_target=='Bream') | (train_target=='Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt= train_target[bream_smelt_indexes]

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(train_bream_smelt,target_bream_smelt)
print(lr.predict(train_bream_smelt[:5]))

print(lr.predict_proba(train_bream_smelt[:5])) # 시그모이드 값

print(lr.coef_,lr.intercept_)

decisions= lr.decision_function(train_bream_smelt[:5])
print(decisions)

from scipy.special import expit

print(expit(decisions))

lr=LogisticRegression(C=20,max_iter=1000)  # C는 규제, max_iter는 반복적인 계산 횟수
lr.fit(train_scaled,train_target)

print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target))

proba=lr.predict_proba(test_scaled[:5])
print(np.round(proba,decimals=3))

print(lr.coef_.shape,lr.intercept_.shape)

# 때에 따라서 sigmoid를 쓰기도하고 softmax를 사용하기도함 결과는 같게 나온다.
decision=lr.decision_function(test_scaled[:5])
print(np.round(decision,decimals=2))

from scipy.special import softmax

proba=softmax(decision,axis=1)
print(np.round(proba,decimals=3))

# 손실함수가 convex(아래로 볼록)라고 가정하고 기울기 방향으로 계수 값을 변화시키는것 = 확률적 경사 하강법
