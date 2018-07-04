
# coding: utf-8

# In[1]:


import librosa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:


def audio_norm(data):
    max_data = np.max(np.absolute(data))
    return data/(max_data+1e-6)*0.5


# In[3]:


class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=4, n_classes=41,
                 use_mfcc=False, n_folds=10, learning_rate=0.0001, 
                 max_epochs=50, n_mfcc=40, datagen_num = 2):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.datagen_num = datagen_num
        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)
        else:
            self.dim = (self.audio_length, 1)


# In[27]:


def prepare_data(df, config, data_dir):
    X = np.empty(shape=(df.shape[0] * config.datagen_num, config.dim[0], config.dim[1], 1))
    y = np.empty(df.shape[0] * config.datagen_num)
    input_length = config.audio_length
    for i, fname in enumerate(df.index):
        print(fname+' ({0}/{1})'.format(i,df.shape[0]))
        file_path = data_dir + fname
        data, _ = librosa.core.load(file_path, sr=config.sampling_rate)
        data = audio_norm(data)
        for j in range(config.datagen_num):
            shifted_data = data
            #pitch shift
            #bpo = 24 #how many steps per octave
            #pr = 3/24 #pitch shift range
            #ps = int(np.random.uniform(-pr * bpo, pr * bpo) + 0.5) #how many (fractional) half-steps to shift y
            #shifted_data = librosa.effects.pitch_shift(shifted_data, config.sampling_rate, n_steps = ps, bins_per_octave = bpo)
            # time stretch
            tr = 1.1 #speed up/down rate
            lgtr = np.log(tr)
            ts = 2 ** np.random.uniform(-lgtr,lgtr)
            shifted_data = librosa.effects.time_stretch(shifted_data, ts)
            #white noise
            #wnvr = 0.05 # white noise volume range
            #wnv  = np.random.uniform(0, wnvr) # white noise volume, random
            #shifted_data += np.random.uniform(-wnv, wnv, shifted_data.shape)
            # Random offset / Padding
            if len(shifted_data) < input_length:
                ratio = input_length/len(shifted_data)
                ratio = np.ceil(ratio)
                shifted_data = np.tile(shifted_data,int(ratio))
            max_offset = len(shifted_data) - input_length
            offset = np.random.randint(max_offset)
            shifted_data = shifted_data[offset:(input_length+offset)]
            #mfcc
            shifted_data = librosa.feature.mfcc(shifted_data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
            shifted_data = np.expand_dims(shifted_data, axis=-1)
            X[ i * config.datagen_num + j, :] = shifted_data
            y[ i * config.datagen_num + j] = df.label_idx[i]
    return X, y


# In[28]:


def prepare_test_data(config, data_dir='../input/audio_test/'):
    df = pd.read_csv('../input/test.csv')
    test_data = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1], 1))
    input_length = config.audio_length
    for i, fname in enumerate(df['fname']):
        print(fname)
        file_path = data_dir + fname
        data, _ = librosa.core.load(file_path, sr=config.sampling_rate)
        if len(data)==0:
            data = np.zeros(88200)
        data = audio_norm(data)
        # Random offset / Padding
        if len(data) < input_length:
            ratio = input_length/len(data)
            ratio = np.ceil(ratio)
            data = np.tile(data,int(ratio))
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
        data = np.expand_dims(data, axis=-1)
        test_data[i] = data
    return test_data


# In[29]:


config = Config(sampling_rate=44100, audio_duration=4, n_folds=5, 
                learning_rate=0.001, use_mfcc=True, n_mfcc=40, datagen_num=5)
train = pd.read_csv("../input/train.csv")
LABELS = list(train.label.unique())
label_idx = {label: i for i, label in enumerate(LABELS)}
train.set_index("fname", inplace=True)
train["label_idx"] = train.label.apply(lambda x: label_idx[x])


# In[33]:


test_data = prepare_test_data(config, data_dir='../input/audio_test/')


# In[ ]:


np.save('../data_gen/test_mfcc.npy',test_data)
del test_data


# In[ ]:


X_train, y = prepare_data(train, config, '../input/audio_train/')


# In[ ]:


np.save('../data_gen/train_mfcc_stretch=1.1_num=5_len=4.npy',X_train)
np.save('../data_gen/label_stretch=1.1_num=5_len=4.npy',y)
