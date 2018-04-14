
# coding: utf-8

# In[2]:


import numpy as np
import sys
import re
import wget
from keras.utils import np_utils
wget.download('http://textfiles.com/etext/AUTHORS/PLATO/plato-cratylus-338.txt', out='DATA.txt')
with open('DATA.txt',encoding='utf8') as f:
    texto_crudo = f.read()


# In[3]:


texto_crudo = texto_crudo.replace('\n','').replace('"','').replace(')','').replace('(','').replace('-','').replace('.','').lower()
chars = sorted(list(set(texto_crudo)))
char_to_int = dict((c,i) for i, c in enumerate(chars))
n_chars = len(texto_crudo)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)


# In[4]:


seq_length = 100
dataX = []
dataY = []
for i in range(0,n_chars - seq_length, 1):
    seq_in = texto_crudo[i:i + seq_length]
    seq_out = texto_crudo[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append([char_to_int[seq_out]])
n_patterns = len(dataX)
print('Total de patrones: ', n_patterns)


# In[5]:


X = np.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)


# In[ ]:


from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1],X.shape[2])))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


# In[ ]:


from keras.callbacks import ModelCheckpoint
nombre="models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(nombre, monitor='loss', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
model.fit(X,y,epochs=50, batch_size=32,callbacks=callbacks_list)
