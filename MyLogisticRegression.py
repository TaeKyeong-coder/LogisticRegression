#모델 생성 및 평가를 위한 sklearn LogisticRegression 사용하기 위해 import
from sklearn.linear_model import LogisticRegression
#학습세트/평가세트 분리를 위한 train_test_split 사용
from sklearn.model_selection import train_test_split
#데이터 정규화를 위한 StandardScaler를 사용
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

#사용할 기반 data 파일 불러오기(제 python버전에서는 encoding값을 안 주면 코드가 안 돌아가서 넣었지만 지워도 됩니다.)
health = pd.read_csv("student_health_3.csv", encoding="cp949")

#data개수 확인 => 1125명의 건강data가 있고 25개의 컬럼이 있음.
print(health.shape)
#잘 불러왔나 확인
print(health.head())

#학년을 도출하기 위한 값으로 수축기, 이완기, 키, 몸무게를 feature로 고르고 데이터 세트 준비.
features = health[['키', '몸무게', '수축기', '이완기']]
grade = health['학년']

#학습세트/평가세트 분리하기
train_features, test_features, train_labels, test_labels = train_test_split(features, grade)

#데이터 정규화(스케일링)하기
scaler = StandardScaler()

train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

#LogisticRegression 모델 생성
model = LogisticRegression()

#features와 labels을 fit시킨다.
model.fit(train_features, train_labels)

#정확도 확인
print(model.score(train_features, train_labels))

#각 features들의 계수(coefficients)를 확인 => 어떤 feature가 학년에 큰 영향을 주는 지
print(model.coef_)

#임의의 데이터(내 data값) 넣어서 학년 예측해보기
TaeGoming = np.array([166.0, 59.0, 80, 52])
sample_health = np.array([TaeGoming])

sample_health = scaler.transform(sample_health)

print(model.predict(sample_health))

#새로운 속성들을 넣었을 때 그 레이블에 속하는지 아닌지 -> 반환은 1또는 0으로 구성된 벡터
#model.predict(features)
#해당 레이블로 분류될 확률 값을 알고 싶을 때 (분류 결과 말고) -> 각 샘플에 대한 확률을 리턴
#model.predict_proba(features)
