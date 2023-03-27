import pandas as pd

fish=pd.read_csv('https://bit.ly/fish_csv_data')

fish_input=fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
fish_target=fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split

train_input,test_input,train_target,test_target=train_test_split(fish_input,fish_target, random_state=42)

from sklearn.preprocessing import StandardScaler

ss=StandardScaler()
ss.fit(train_input)

train_scaled=ss.transform(train_input)
test_scaled=ss.transform(test_input)

print(len(train_scaled))
print(len(test_scaled))

from sklearn.linear_model import SGDClassifier

sc=SGDClassifier(loss='log', max_iter=10,random_state=42)
sc.fit(train_scaled,train_target)
print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled,test_target))

sc.partial_fit(train_scaled,train_target) # 훈련한 모델 sc를 추가 훈련함 이어서 훈련시킬때 partial.fit사용
print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled,test_target))


import numpy as np

sc=SGDClassifier(loss='log',random_state=42)
train_score=[]
test_score=[]
classes=np.unique(train_target) # train_target에 있는 7개의 생선 목록 생성.

for i in range(0,300):
    sc.partial_fit(train_scaled,train_target,classes=classes)
    train_score.append(sc.score(train_scaled,train_target))
    test_score.append(sc.score(test_scaled,test_target))

import matplotlib.pyplot as plt

print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled,test_target))
plt.plot(train_score)
plt.plot(test_score)
plt.show()

sc=SGDClassifier(loss='hinge',random_state=42, max_iter=100, tol=None)
sc.fit(train_scaled,train_target)
print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled,test_target))
