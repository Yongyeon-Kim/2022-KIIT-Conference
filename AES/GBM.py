import pandas as pd
import numpy as np

# 데이터 로드 
DATA_IN_PATH = 'D:/Users/Yongyeon/python/2022_학술대회\Random Forest/dataset/'
# TRAIN_CLEAN_DATA = 'essay_1_total.csv'
# TRAIN_CLEAN_DATA = 'essay_2_total.csv'
# TRAIN_CLEAN_DATA = 'essay_3_total.csv'
# TRAIN_CLEAN_DATA = 'essay_4_total.csv'
TRAIN_CLEAN_DATA = 'essay_5_total.csv'
# TRAIN_CLEAN_DATA = 'essay_6_total.csv'

train_data = pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)
reviews = list(train_data['sentence'])
y = np.array(train_data['label'])

#벡터화
from sklearn.feature_extraction.text import CountVectorizer 
vectorizer = CountVectorizer(analyzer='word', max_features=512)
train_data_features = vectorizer.fit_transform(reviews)

# 학습과 검증 데이터 분리
from sklearn.model_selection import train_test_split
TEST_SIZE = 0.2
RANDOM_SEED = 42

train_input, eval_input, train_label, eval_label = \
    train_test_split(train_data_features, y, test_size=TEST_SIZE, random_state=1)
    

# 모델 구현 및 학습
from sklearn.ensemble import GradientBoostingClassifier

for _ in range(10):
    #랜덤 포레스트 분류기에 100개의 의사결정 트리를 사용합니다.
    forest = GradientBoostingClassifier(n_estimators=100)

    # 단어 묶음을 벡터화한 데이터와 정답 데이터를 가지고 학습을 시작한다.
    forest.fit(train_input, train_label)

    # 검증 데이터셋으로 성능 평가
    print("Accuracy: %f" % forest.score(eval_input, eval_label))