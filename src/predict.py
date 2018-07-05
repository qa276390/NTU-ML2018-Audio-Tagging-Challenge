
# In[78]:


# Change this to True to replicate the result
COMPLETE_RUN = True

import numpy as np
np.random.seed(1001)

import os
import shutil
from keras.models import load_model

#import IPython
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#from tqdm import tqdm_notebook
#from sklearn.cross_validation import StratifiedKFold
import tensorflow as tf
import wave
print('tf:',tf.__version__)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

#get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')
os.getpid()


# In[80]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/sample_submission.csv")


train['nframes'] = train['fname'].apply(lambda f: wave.open('../input/audio_train/' + f).getnframes())
test['nframes'] = test['fname'].apply(lambda f: wave.open('../input/audio_test/' + f).getnframes())

# In[94]:


import librosa
import numpy as np
import scipy
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D, 
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)
from keras.utils import Sequence, to_categorical
import keras
print('keras:',keras.__version__)


# <a id="configuration"></a>
# #### Configuration

# The Configuration object stores those learning parameters that are shared between data generators, models, and training functions. Anything that is `global` as far as the training is concerned can become the part of Configuration object.

# In[95]:


class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=2, n_classes=41,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001, 
                 max_epochs=50, n_mfcc=20):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)


# In[96]:


class DataGenerator(Sequence):
    def __init__(self, config, data_dir, list_IDs, labels=None, 
                 batch_size=64, preprocessing_fn=lambda x: x):
        self.config = config
        self.data_dir = data_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.preprocessing_fn = preprocessing_fn
        self.on_epoch_end()
        self.dim = self.config.dim

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        cur_batch_size = len(list_IDs_temp)
        X = np.empty((cur_batch_size, *self.dim))

        input_length = self.config.audio_length
        for i, ID in enumerate(list_IDs_temp):
            file_path = self.data_dir + ID
            
            # Read and Resample the audio
            data, _ = librosa.core.load(file_path, sr=self.config.sampling_rate,
                                        res_type='kaiser_fast')


            
            # Random offset / Padding
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length+offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
                
            # Normalization + Other Preprocessing
            if self.config.use_mfcc:
                data = librosa.feature.mfcc(data, sr=self.config.sampling_rate,
                                                   n_mfcc=self.config.n_mfcc)
                data = np.expand_dims(data, axis=-1)
            else:
                data = self.preprocessing_fn(data)[:, np.newaxis]
            X[i,] = data

        if self.labels is not None:
            y = np.empty(cur_batch_size, dtype=int)
            for i, ID in enumerate(list_IDs_temp):
                y[i] = self.labels[ID]
            return X, to_categorical(y, num_classes=self.config.n_classes)
        else:
            return X


# <a id="1d_normalization"></a>
# #### Normalization
# 
# Normalization is a crucial preprocessing step. The simplest method is rescaling the range of features to scale the range in [0, 1]. 

# In[97]:



def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5



LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}
train.set_index("fname", inplace=True)
test.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: label_idx[x])

config = Config(sampling_rate=16000, audio_duration=2, n_folds=10, learning_rate=0.001, max_epochs=100)



# In[ ]:


#```python
PREDICTION_FOLDER = "predictions_1d_conv"
if not os.path.exists(PREDICTION_FOLDER):
    os.mkdir(PREDICTION_FOLDER)
#if os.path.exists('logs/' + PREDICTION_FOLDER):
#    shutil.rmtree('logs/' + PREDICTION_FOLDER)

#skf = StratifiedKFold(train.label_idx, n_folds=config.n_folds)

for i in range(10):
    print("Fold: ", i)
    print("#"*50)
    
    model = load_model('model_1d/best_%d.h5'%i)
    
    
    # Save test predictions
    test_generator = DataGenerator(config, '../input/audio_test/', test.index, batch_size=128,
                                    preprocessing_fn=audio_norm)
    predictions = model.predict_generator(test_generator, use_multiprocessing=True, 
                                          workers=6, max_queue_size=20, verbose=1)
    np.save(PREDICTION_FOLDER + "/test_predictions_%d.npy"%i, predictions)
    
    # Make a submission file
    top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test['label'] = predicted_labels
    test[['label']].to_csv(PREDICTION_FOLDER + "/predictions_%d.csv"%i)
    #```


# In[101]:


from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation)
from keras.utils import Sequence, to_categorical
from keras import backend as K


# In[102]:



# In[103]:


config = Config(sampling_rate=44100, audio_duration=2, n_folds=10, 
                learning_rate=0.001, use_mfcc=True, n_mfcc=40)
if not COMPLETE_RUN:
    config = Config(sampling_rate=44100, audio_duration=2, n_folds=2, 
                    max_epochs=1, use_mfcc=True, n_mfcc=40)


# In[104]:


# 
X_test = np.load('../input/mfcc/test_mfcc.npy')
#X_test = np.load('../input/mfcc/test_mfcc_len=4.npy')


# mean = np.mean(X_train, axis=0)
# std = np.std(X_train, axis=0)
# 
# X_train = (X_train - mean)/std
# X_test = (X_test - mean)/std
# 
# 
PREDICTION_FOLDER = "predictions_2d_conv"
if not os.path.exists(PREDICTION_FOLDER):
    os.mkdir(PREDICTION_FOLDER)

for i in range(10):
    #K.clear_session()
    #X, y, X_val, y_val = X_train[train_split], y_train[train_split], X_train[val_split], y_train[val_split]
    #checkpoint = ModelCheckpoint('model/best_%d.h5'%i, monitor='val_loss', verbose=1, save_best_only=True)
    #early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    #tb = TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold_%i'%i, write_graph=True)
    #callbacks_list = [checkpoint, early, tb]
    print("#"*50)
    print("Fold: ", i)
    #model = get_2d_conv_model(config)
    #history = model.fit(X, y, validation_data=(X_val, y_val), callbacks=callbacks_list, batch_size=64, epochs=config.max_epochs)
    model = load_model('model_2d/best_%d.h5'%i)
# 
    # Save train predictions
    #predictions = model.predict(X_train, batch_size=64, verbose=1)
    #np.save(PREDICTION_FOLDER + "/train_predictions_%d.npy"%i, predictions)
# 
    # Save test predictions
    predictions = model.predict(X_test, batch_size=64, verbose=1)
    np.save(PREDICTION_FOLDER + "/test_predictions_%d.npy"%i, predictions)
# 
    # Make a submission file
    top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test['label'] = predicted_labels
    test[['label']].to_csv(PREDICTION_FOLDER + "/predictions_%d.csv"%i)
# 


# In[105]:


import numpy as np


PREDICTION_FOLDER_1 = "predictions_1d_conv"
PREDICTION_FOLDER_2 = "predictions_2d_conv"

# In[107]:


pred_list = []
for i in range(10):
    pred_list.append(np.load(PREDICTION_FOLDER_1 + "/test_predictions_%d.npy"%i))
for i in range(10):
    pred_list.append(np.load(PREDICTION_FOLDER_2 + "/test_predictions_%d.npy"%i))

#prediction = np.ones_like(pred_list[0])
predlog = np.zeros_like(pred_list[0])
for pred in pred_list:
    #prediction = prediction*pred
    predlog = predlog + np.log(pred)
    
#prediction = prediction**(1./len(pred_list))
predlog = predlog / (len(pred_list))


#using log
prediction = predlog

# Make a submission file
top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
predicted_labels = [' '.join(list(x)) for x in top_3]
test = pd.read_csv('../input/sample_submission.csv')
test['label'] = predicted_labels
test[['fname', 'label']].to_csv("1d_2d_ensembled_submission.csv", index=False)


