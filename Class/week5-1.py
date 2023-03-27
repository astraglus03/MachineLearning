import pandas as pd # 대량 데이터 처리 할때 주로 사용
import numpy as np
from sklearn.model_selection import train_test_split # 학습용 실험용
from sklearn.preprocessing import StandardScaler # 표준


fish = pd.read_csv('https://bit.ly/fish_csv')  # 데이터 불러오기
fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
# 5개 정보 가져오기(판다스로) numpy규격으로 바꾸는게 편함
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
ss = StandardScaler() #표준 점수화
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

from sklearn.linear_model import SGDClassifier

sc= SGDClassifier(loss='log', max_iter=5,random_state=42)
sc.fit(train_scaled,train_target)

print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled,test_target))

sc.partial_fit(train_scaled,train_target)

print(sc.score(train_scaled,train_target))

print(sc.score(test_scaled,test_target))

sc= SGDClassifier(loss='log', random_state=42)
train_score=[]
test_score=[]

classes=np.unique(train_target)
for _ in range(0,300):
    sc.partial_fit(train_scaled,train_target,classes=classes)
    # 조기종료 옵션 주는이유: 애초에 주머니가 어떻게
    # 생겼는지 모르기때문에 일단 돌려보고 본다
    train_score.append(sc.score(train_scaled,train_target))
    test_score.append(sc.score(test_scaled,test_target))

sc=SGDClassifier(loss='log', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled,train_target)

print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled,test_target))
