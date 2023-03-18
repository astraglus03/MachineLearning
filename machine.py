import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = np.load('data_input.npy')  # 데이터 로드. @파일명
target = np.load('data_target.npy')

plt.scatter(data[:500,0],data[:500,1])
plt.scatter(data[500:900,0],data[500:900,1])
plt.scatter(data[900:1300,0],data[900:1300,1])
plt.scatter(data[1300:,0],data[1300:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn = KNeighborsClassifier()
input_arr = np.array(data)
target_arr = np.array(target)

index = np.arange(1800)
np.random.seed(42)
np.random.shuffle(index)

train_input = input_arr[index[:1200]]
train_target = target_arr[index[:1200]]

test_input = input_arr[index[1200:]]
test_target = target_arr[index[1200:]]

# train_input, test_input, train_target, test_target = \
#     train_test_split(data, target, stratify=target, random_state=42)
# print(train_input.shape, test_input.shape)
# print(train_target.shape, test_target.shape)

kn = kn.fit(train_input, train_target)
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

print(mean, std)
train_scaled = (train_input - mean) / std
new = ([23, 400] - mean) / std
kn.fit(train_scaled, train_target)
plt.subplot(221)

distance, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0],new[1], marker='^')
plt.scatter(train_scaled[indexes, 0],train_scaled[indexes,1],marker='D')
plt.subplot(222)

kn.n_neighbors =10
distance, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0],new[1], marker='^')
plt.scatter(train_scaled[indexes, 0],train_scaled[indexes,1],marker='D')
plt.subplot(223)

kn.n_neighbors =30
distance, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0],new[1], marker='^')
plt.scatter(train_scaled[indexes, 0],train_scaled[indexes,1],marker='D')
plt.subplot(224)

kn.n_neighbors =50
distance, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0],new[1], marker='^')
plt.scatter(train_scaled[indexes, 0],train_scaled[indexes,1],marker='D')
plt.show()
