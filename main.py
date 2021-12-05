import numpy as np
import pandas as pd
from clenar import stopword, clean1
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from random import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.models import Model
from keras.layers import BatchNormalization
from sklearn.metrics import f1_score
from gensim.models import KeyedVectors
import gensim
import json
import warnings
import logging, sys
import os
from datetime import datetime
logging.disable(sys.maxsize)
warnings.filterwarnings('ignore')




  

train_data = pd.read_csv('./data/train_data.csv')
train_data.drop('path', inplace=True, axis=1)
valid_data = pd.read_csv('./data/valid_data.csv')
valid_data.drop('path', inplace=True, axis=1)
# Here we can use pd.Categorical(valid_data['action']) 

act_encoder = OneHotEncoder(sparse=False, dtype='int32', handle_unknown='ignore')
obj_encoder = OneHotEncoder(sparse=False, dtype='int32', handle_unknown='ignore')
loc_encoder = OneHotEncoder(sparse=False, dtype='int32', handle_unknown='ignore')

act_oh = act_encoder.fit_transform(np.array(train_data['action']).reshape(-1,1))
obj_oh = obj_encoder.fit_transform(np.array(train_data['object']).reshape(-1,1))
loc_oh = loc_encoder.fit_transform(np.array(train_data['location']).reshape(-1,1))
train_data['act'] = list(act_oh)
train_data['obj'] = list(obj_oh)
train_data['loc'] = list(loc_oh)
train_data.head()

train_data['transcription'] = train_data['transcription'].apply(clean1)
valid_data['transcription'] = valid_data['transcription'].apply(stopword)
valid_data['transcription'] = valid_data['transcription'].apply(clean1)
train_data['transcription'] = train_data['transcription'].apply(stopword)

train_data.head()

vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train_data['transcription'])
token_train = tokenizer.texts_to_sequences(train_data['transcription'])
token_valid = tokenizer.texts_to_sequences(valid_data['transcription'])
vocab_size = len(tokenizer.word_index) + 1
# As the length of arrays are diffrent we need to pad it
max_length = 200
train_data['padded_sequences'] = pad_sequences(token_train,  padding='post', maxlen=max_length).tolist()
train = np.array([np.array(i) for i in train_data['padded_sequences']])
valid_data['padded_sequences'] = pad_sequences(token_valid,  padding='post', maxlen=max_length).tolist()
valid = np.array([np.array(i) for i in train_data['padded_sequences']])

#For word2vec
from gensim.models import KeyedVectors
e_model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index)+1, 300))
for word, i in word_index.items():
    try:    
      embedding_vector = e_model[word]
      if embedding_vector is not None:
          embedding_matrix[i] = embedding_vector
    except:
      embedding_matrix[i] = np.zeros((300,))
      print(1)

# Treating [location,object,action] as class and training the model on them
t = pd.DataFrame()
t["all"] =train_data['act']
for i in range(len(train_data)):
  t['all'].iloc[i]= list(train_data['act'].iloc[i])+list(train_data['obj'].iloc[i])+list(train_data['loc'].iloc[i])
Y = pd.DataFrame()
for i in range(23):
  Y[i+1]=t['all'].apply(lambda x: int(x[i]))
# Taking X as the padded output of the tokenier
X = train
#seperating the data in test and train pair
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

#Model
EMBEDDING_DIM = 300
embedding_layer = Embedding(vocab_size,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=10,
        trainable=False)

lstm_layer_1 = LSTM(20, return_sequences=False, dropout=0.15, recurrent_dropout=0.15)

sequence_input = Input(shape=(200,), dtype='int32')
x1 = embedding_layer(sequence_input)
x1 = lstm_layer_1(x1)
x1 = Dropout(0.10)(x1)
x1 = BatchNormalization()(x1)
x1 = Dense(500, activation='relu')(x1)
x1 = Dropout(0.10)(x1)
x1 = Dense(50, activation='relu')(x1)
x1 = Dropout(0.10)(x1)
x1 = BatchNormalization()(x1)
preds = Dense(23, activation='softmax')(x1)

model = Model(inputs=sequence_input, outputs=preds)
print(model.summary())


loss_fn = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
train_acc_metric = tf.keras.losses.MeanAbsoluteError()
val_acc_metric = tf.keras.losses.MeanAbsoluteError()

model.compile(optimizer=optimizer, loss=loss_fn, metrics=train_acc_metric)
model.fit(X_train, y_train, batch_size=1024, epochs=1000, validation_data=(X_test, y_test), use_multiprocessing=True)


test_df= valid_data

tokenizer = Tokenizer()
tokenizer.fit_on_texts(test_df['transcription'])

max_len = 200
data = tokenizer.texts_to_sequences(test_df['transcription'].values)
data = pad_sequences(data, maxlen=max_len)

preds = model.predict(data)
ans=preds
# print(ans)

test_df['act']=test_df['transcription']
test_df['obj']=test_df['transcription']
test_df['loc']=test_df['transcription']

for i in range(len(test_df)):
  temp=[]
  for j in range(6):
    temp.append(round(ans[i][j]))
  test_df['act'].iloc[i]= np.array(temp)

  temp=[]
  for j in range(4,18):
    temp.append(round(ans[i][j]))
  test_df['obj'].iloc[i]= np.array(temp)
  
  temp=[]
  for j in range(19,23):
    temp.append(round(ans[i][j]))
  test_df['loc'].iloc[i]= np.array(temp)

# print(test_df.head())

# test_df.head()

test_df['act_pred']= test_df['act'].apply(lambda x: act_encoder.inverse_transform(x.reshape(1,-1)))
test_df['obj_pred']= test_df['obj'].apply(lambda x: obj_encoder.inverse_transform(x.reshape(1,-1)))
test_df['loc_pred']= test_df['loc'].apply(lambda x: loc_encoder.inverse_transform(x.reshape(1,-1)))

test_df['act_pred']=test_df['act_pred'].apply(lambda x: x[0][0])
test_df['act_pred']=test_df['act_pred'].apply(lambda x: x if x else 'none')

test_df['obj_pred']=test_df['obj_pred'].apply(lambda x: x[0][0])
test_df['obj_pred']=test_df['obj_pred'].apply(lambda x: x if x else 'none')

test_df['loc_pred']=test_df['loc_pred'].apply(lambda x: x[0][0])
test_df['loc_pred']=test_df['loc_pred'].apply(lambda x: x if x else 'none')

# print(test_df['act_pred'].head())
# micro f1 score
action_f1 = f1_score(test_df['act_pred'],test_df['action'], average='micro')
object_f1 = f1_score(test_df['obj_pred'],test_df['object'], average='micro')
location_f1 = f1_score(test_df['loc_pred'],test_df['location'], average='micro')
print('F1 score for action-->',action_f1)
print('F1 score for object-->',object_f1)
print('F1 score for location-->',location_f1)