#앞에서 만들었던거 사용함
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=42)

print(x_train.shape, x_val.shape) #훈련 전 데이터의 크기를 확인하는 습관을 가지자.

#사이킷런
#사이킷런에는 분류작업을 위한 MLPClassifier, 회귀작업을 위한 MLPRegressor가 있음.
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=(10, ), activation='logistic', solver='sgd', alpha=0.01, batch_size=32, learning_rate_init=0.1, max_iter=500)
#hidden_layer_sizes 은닉층의 크기를 정함. 10개의 뉴런을 가진 2개의 은닉층을 만들려면 (10,10)으로 입력함. 여기서는 (10, )이므로 10개의 뉴런을 가진 하나의 은닉층
#activation은 활성화함수를 정함. 사이킷런에서는 은닉층마다 다른 함수로 할 수는 없고 다 같은 활성화함수를 씀.
#solver는 경사하강법 알고리즘의 종류를 정함. sgd는 확률적 경사 하강법임.
#alpha는 규제 적용을 위한 매개변수임. l2값을 지정함.
#batch_size는 배치의 크기를 정함. (미니배치경사하강법)
mlp.fit(x_train_scaled, y_train)
mlp.score(x_val_scaled, y_val)
