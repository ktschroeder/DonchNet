import os
import feature_extraction.map_json_to_feats as jtf
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Conv2D, Dense, Flatten, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from tensorflow.keras.activations import tanh, elu, relu
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence


# mapFeats = []
# songFeats = []
mainDir = "data/stored_feats"

def getMapFeats():
    mapFeats = []
    for dir in os.listdir(mainDir):
        for item in os.listdir(os.path.join(mainDir, dir)):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".json":
                id, offsets = jtf.jsonToFeats(os.path.join(mainDir, dir, item))
                mapFeats.append([id, offsets])
    return mapFeats

def getSongFeats():
    songFeats = []
    for dir in os.listdir(mainDir):
        for item in os.listdir(os.path.join(mainDir, dir)):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".pkl":
                file = open(os.path.join(mainDir, dir, item), 'rb')
                data = pickle.load(file)
                songFeats.append([item[:-4], data])  # the -4 removes ".pkl"
    return songFeats

def jointlyGetMapAndSongFeats():
    # feats[]  # stored jointly for convenience: there may be multiple mapFeats per songFeats
    mapFeats = []
    songFeats = []
    index = 0
    for dir in os.listdir(mainDir):
        mapFeats.append([])
        songFeats.append([])
        for item in os.listdir(os.path.join(mainDir, dir)):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".json":
                id, onsets = jtf.jsonToFeats(os.path.join(mainDir, dir, item))
                mapFeats[index].append([id, onsets])
            if ext == ".pkl":
                file = open(os.path.join(mainDir, dir, item), 'rb')
                data = pickle.load(file)
                songFeats[index].append([item[:-4], data])  # the -4 removes ".pkl"
        index += 1

    return mapFeats, songFeats

def prepareFeatsForModel(mapFeats, songFeats):
    pMapFeats = []
    pSongFeats = []
    idMap = -1
    
    for i in range(len(mapFeats)):
        audioFrames = len(songFeats[i][0][1])

        # number of frames should line up with frames in audio feats
        for map in mapFeats[i]:
            # temp = 0
            idMap += 1
            pMapFeats.append([])
            onsets = map[1]  # time in ms of objects in map
            hitIndex = 0  # index of hitObject in the map (listed in variable: onsets)
            groundTruth = 0  # whether there is an object in this frame
            for j in range(audioFrames * 10):  # for each millisecond
                
                if hitIndex < len(onsets) and int(onsets[hitIndex]) <= j:  # if there is a hit
                    hitIndex += 1
                    groundTruth = 1
                if j % 10 == 9:  # at every 10 milliseconds we summarize the 10ms frame and reset
                    pMapFeats[idMap].append(groundTruth)
                    # temp += groundTruth
                    groundTruth = 0
            # print(temp, "hits")
        
        #  number of maps using the same audio file
        songRepeats = len(mapFeats[i])   # for now I am neglecting the concern about multiple maps sharing a song beind dispersed between training and test sets
        for j in range(songRepeats):
            pSongFeats.append(songFeats[i][0][1])

    return pMapFeats, pSongFeats

def createModel(mapFeats, songFeats):
    # model = keras.Sequential()
    # # Add an Embedding layer expecting input vocab of size 1000, and
    # # output embedding dimension of size 64.
    # model.add(layers.Embedding(input_dim=1000, output_dim=64))

    # # Add a LSTM layer with 128 internal units.
    # model.add(layers.LSTM(128))

    # # Add a Dense layer with 10 units.
    # model.add(layers.Dense(10))

    # model.summary()
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=1000, output_dim=64))

    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(layers.GRU(256, return_sequences=True))

    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    model.add(layers.SimpleRNN(128))

    model.add(layers.Dense(10))

    model.summary()

    model.compile()  # TODO ensure my GPU is being used
    print("compiled model")

def createConvLSTM():
    #nn = models.Sequential()
    # nn.add(layers.Conv3D(?, activation = 'relu', input_dim=?)) # 
    #nn.add(layers.Conv)

    # FULL CNN MODEL
    # convolutional layer with 10 filter kernels that are 7-wide in time and 3-wide in frequency. ReLU.
    # 1D max-pooling only in the frequency dimension, with a width and stride of 3
    # convolutional layer with 20 filter kernels that are 3-wide in time and 3-wide in frequency. ReLU.
    # 1D max-pooling only in the frequency dimension, with a width and stride of 3
    # fully connected layerr with ReLU activation functions and 256 nodes
    # fully connected layerr with ReLU activation functions and 128 nodes

    # C-LSTM model
    # convolutional layer with 10 filter kernels that are 7-wide in time and 3-wide in frequency. ReLU.
    # 1D max-pooling only in the frequency dimension, with a width and stride of 3
    # convolutional layer with 20 filter kernels that are 3-wide in time and 3-wide in frequency. ReLU.
    #? 1D max-pooling only in the frequency dimension, with a width and stride of 3
    # Output of second conv. layer is a 2D tensor for me. Flatten it along the (channel and) frequency axis. (preserving temporal dimension)
    # LSTM with 200 nodes
    # LSTM with 200 nodes
    # fully connected ReLU layer with dimension 256
    # fully connected ReLU layer with dimension 128
    # this model is trained using 100 unrollings for backpropagation through time.

    learning_rate = 0.01
    conv1d_strides = 12
    conv1d_1_strides = 12   
    conv1d_filters = 4
    hidden_units = 24
    # Create Sequential Model ###########################################
    clear_session()
    model = Sequential()
    model.add(Conv2D(conv1d_filters, (7,3),strides=conv1d_strides, activation=None, padding='same',input_shape=(25,40,1))) # shape is 12+1+12 frames x 40 frequency bands
    model.add(Conv2D(conv1d_filters, (7,3),strides=conv1d_strides, activation=None, padding='same'))
    model.add(TimeDistributed(Flatten())) # see above notes, does this overly flatten temporal?
    model.add(LSTM(hidden_units))
    model.add(Dense(1, activation=None))
    model.compile(optimizer=Adam(learning_rate=learning_rate))# loss=error_to_signal, metrics=[error_to_signal])
    print(model.summary())

    print()

def basicModel():
    # mapFeats = getMapFeats()
    # songFeats = getSongFeats()  # array of [mels pkl title, mels pkl data] pairs, one per song folder
    mapFeats, songFeats = jointlyGetMapAndSongFeats()
    mapFeats, songFeats = prepareFeatsForModel(mapFeats, songFeats)
    # mapFeats = sorted(mapFeats, key=lambda x: x[0])  # sort by id (makeshift title: song-mapper-diff)
    # songFeats = sorted(songFeats, key=lambda x: x[0])  # similar to above
    # no need to do above sorting with new implementation, they already line up.
    # sloppy but at this point we expect the IDs to basically line up. But notably there are usually more mapFeats than songFeats: Some maps share a song.
    # print(mapFeats[8][0][0])
    # print(songFeats[8][0][0])

    for i in range(14):
        print(len(mapFeats[i]), len(songFeats[i]))
    # print(songFeats[0][0][1].shape)
    # createConvLSTM()

    # createModel(mapFeats, songFeats)
    # print(songFeats[0][1][82])
    # print(mapFeats.shape)
    # print()
    # print(songFeats[0][1].shape)

basicModel()
# createConvLSTM()