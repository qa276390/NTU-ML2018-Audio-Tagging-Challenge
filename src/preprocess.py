import sys
import numpy as np
import pandas as pd
import librosa
from keras.models import Sequential, load_model
from keras.layers import Input, Embedding, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence

def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    max = max_data
    if max < -min_data: 
        max = -min_data
    data = data/(max+1e-6)
    return data * 0.5

train = pd.read_csv("data/train.csv")
#test = pd.read_csv("data/sample_submission.csv")
#print(train.fname[0])
data, sr = librosa.core.load("data/audio_train/"+train.fname[0], sr=16000,res_type='kaiser_fast')

# normalize volume to -0.5 ~ 0.5
data = audio_norm(data)

# time stretch
tr   = 2.0 # time stretch range
lgtr = np.log2(tr)
ts   = 2 ** np.random.uniform(-lgtr, lgtr)
data = librosa.effects.time_stretch(data, ts) #rate=2.0

# pitch shift
bpo  = 24 # bins_per_octave
pr   = 1  # pitch shift range (in octave)
ps   = int(np.random.uniform(-pr * bpo, pr * bpo) + 0.5)
data = librosa.effects.pitch_shift(data, sr, n_steps=ps, bins_per_octave=24)

# white noise
wnvr = 0.1 # white noise volume range
wnv  = np.random.uniform(0, wnvr) # white noise volume, random
data += np.random.uniform(-wnv, wnv, data.shape)

# output 
librosa.output.write_wav('output.wav', data, sr)


class Config():
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

            # time stretch
            data = librosa.effects.time_stretch(data, 2.0) #rate=2.0
            #data = librosa.effects.time_stretch(data, 0.5)

            # Random offset / Padding
            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.random(0, max_offset)
                data = data[offset:(input_length+offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.random(0, max_offset)
                else:
                    offset = 0
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
                
            # Normalization + Other Preprocessing
            #Shift down by a tritone (six half-steps)
            data = librosa.effects.pitch_shift(data, sr, n_steps=-6, bins_per_octave=12) 
            #Shift up by 3 quarter-tones
            #data = librosa.effects.pitch_shift(data, sr, n_steps=3, bins_per_octave=24)

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

