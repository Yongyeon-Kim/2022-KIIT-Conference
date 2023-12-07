import pandas as pd
import numpy as np

import re
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

from keras.optimizers import Adam, RMSprop

# 데이터 로드 
DATA_IN_PATH = 'D:/Users/Yongyeon/python/2022_학술대회\Random Forest/dataset/'
# TRAIN_CLEAN_DATA = 'essay_1_total.csv'
TRAIN_CLEAN_DATA = 'essay_2_total.csv'
# TRAIN_CLEAN_DATA = 'essay_3_total.csv'
# TRAIN_CLEAN_DATA = 'essay_4_total.csv'
# TRAIN_CLEAN_DATA = 'essay_5_total.csv'
# TRAIN_CLEAN_DATA = 'essay_6_total.csv'

train_data = pd.read_csv(DATA_IN_PATH + TRAIN_CLEAN_DATA)
labels_ = pd.unique(train_data.label)
labels_ = labels_.tolist()
labels_.sort()
label_range = labels_[-1] - labels_[0] + 2
sentences = np.array(train_data['sentence'])
labels = np.array(train_data['label'])

max_len = 512

from sklearn.model_selection import train_test_split
TEST_SIZE = 0.2
RANDOM_SEED = 42
# 42 40 38 36 34 32 30

train_input, eval_input, train_label, eval_label = \
        train_test_split(sentences, labels, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_input)
vocab_size = len(tokenizer.word_counts)
print("vocabulary size: ", vocab_size)

train_input = tokenizer.texts_to_sequences(train_input)
eval_input  = tokenizer.texts_to_sequences(eval_input)

train_input = pad_sequences(train_input, maxlen=max_len)
eval_input = pad_sequences(eval_input, maxlen=max_len)

model = Sequential([
    Embedding(vocab_size+1, 64),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(label_range, activation='softmax')
])
model.summary()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model15.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)

model.compile(optimizer=rmsprop, 
              loss='sparse_categorical_crossentropy', 
              metrics=['acc'])
model.summary()

history = model.fit(train_input, 
                    train_label,
                    epochs=20, 
                    callbacks=[es, mc], 
                    batch_size=1, 
                    validation_split=0.2)

loaded_model = load_model('best_model15.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(eval_input, eval_label)[1]))