import os
import feature_extraction.map_json_to_feats as jtf
import pickle
import numpy as np
from numpy import *
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
    mapCount = 0
    for dir in os.listdir(mainDir):
        mapFeats.append([])
        songFeats.append([])
        for item in os.listdir(os.path.join(mainDir, dir)):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".json":
                id, onsets = jtf.jsonToFeats(os.path.join(mainDir, dir, item))
                mapFeats[index].append([id, onsets])
                mapCount += 1
            if ext == ".pkl":
                file = open(os.path.join(mainDir, dir, item), 'rb')
                data = pickle.load(file)
                songFeats[index].append([item[:-4], data])  # the -4 removes ".pkl"
        index += 1

    return mapFeats, songFeats, mapCount

max_sequence_length = 30000
def prepareFeatsForModel(mapFeats, songFeats, mapCount):
    pMapFeats = np.full((mapCount, max_sequence_length), -999)
    pSongFeats = np.full((mapCount, max_sequence_length, 40), -999)
    idMap = -1
    idSong = -1
    
    for i in range(len(mapFeats)):
        audioFrames = len(songFeats[i][0][1])

        # number of frames should line up with frames in audio feats
        for map in mapFeats[i]:
            # temp = 0
            idMap += 1
            onsets = map[1]  # time in ms of objects in map
            hitIndex = 0  # index of hitObject in the map (listed in variable: onsets)
            groundTruth = 0  # whether there is an object in this frame
            groundTruths = []
            for j in range(audioFrames * 10):  # for each millisecond
                if j / 10 >= max_sequence_length:
                    break
                if hitIndex < len(onsets) and int(onsets[hitIndex]) <= j:  # if there is a hit
                    hitIndex += 1
                    groundTruth = 1
                if j % 10 == 9:  # at every 10 milliseconds we summarize the 10ms frame and reset
                    groundTruths.append(groundTruth)
                    # temp += groundTruth
                    groundTruth = 0
            # print(temp, "hits")
            for j in range(len(groundTruths), max_sequence_length):
                groundTruths.append(-999)
            # pMapFeats.append(tf.convert_to_tensor(groundTruths, dtype=tf.int32))
            pMapFeats[idMap] = groundTruths

        #  number of maps using the same audio file
        songRepeats = len(mapFeats[i])   # for now I am neglecting the concern about multiple maps sharing a song beind dispersed between training and test sets
        for j in range(songRepeats):
            idSong += 1
            # pSongFeats.append(tf.convert_to_tensor(songFeats[i][0][1], dtype=tf.int32))
            trimmedSongFeats = songFeats[i][0][1][:min(len(songFeats[i][0][1]), max_sequence_length)]  # trim to max sequence length if applicable
            pad = []
            for k in range (max_sequence_length - len(trimmedSongFeats)):
                # print("got in with", max_sequence_length - len(trimmedSongFeats))
                pad.append(np.full((40), -999))
            # print(pad)
            if(len(pad) > 0):
                trimmedSongFeats = np.vstack((trimmedSongFeats, pad))
            pSongFeats[idSong] = trimmedSongFeats

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
    mapFeats, songFeats, count = jointlyGetMapAndSongFeats()
    y, x = prepareFeatsForModel(mapFeats, songFeats, count)
    #nn = models.Sequential()
    # nn.add(layers.Conv3D(?, activation = 'relu', input_dim=?)) # 
    #nn.add(layers.Conv)
    # print(np.shape(mapFeats[0][0]))

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
    padding_value = -999
    seq_length_cap = 30000  # 30000 frames = 300 seconds = 5 minutes

    # xpad = np.full((17, seq_length_cap, 3), fill_value=padding_value)
    # for s, x in enumerate(songFeats):
    #     seq_len = x.shape[0]
    #     xpad[s, 0:seq_len, :] = x
    # print(xpad)

    x = x.reshape(len(x),len(x[0]),len(x[0][0]),1)  # 17, 30000, 40, 1
    y = y.reshape(len(y),len(y[0]),1)   

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


    # indices = tf.range(start=0, limit=tf.shape(songFeats)[0], dtype=tf.int32)
    # shuffled_indices = tf.random.shuffle(indices)

    # X_random = tf.gather(songFeats, shuffled_indices)
    # Y_random = tf.gather(mapFeats, shuffled_indices)
    # songFeats = tf.convert_to_tensor(songFeats, dtype=tf.int32)
    # x = songFeats.reshape(len(songFeats),1)

    # mapFeats = np.asarray(mapFeats).astype('int32')
    # y = mapFeats.reshape(len(mapFeats), 1)

    test_size = 0.2
    epochs = 5
    batch_size = 3 # ?
    # shuffled_indices = np.random.permutation(len(songFeats)) 
    # X_random = tf.gather(songFeats, shuffled_indices)
    # y_random = tf.gather(mapFeats, shuffled_indices)

    # mapFeats = tf.ragged.constant(mapFeats)

    model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=test_size)
    # model.fit(np.asarray(songFeats).astype('float32'), np.asarray(mapFeats).astype('float32'), epochs=epochs, batch_size=batch_size, validation_split=test_size)
    #model.save('test_model.h5')

    print("got to end")

def basicModel():
    # mapFeats = getMapFeats()
    # songFeats = getSongFeats()  # array of [mels pkl title, mels pkl data] pairs, one per song folder
    mapFeats, songFeats, totalMaps = jointlyGetMapAndSongFeats()
    mapFeats, songFeats = prepareFeatsForModel(mapFeats, songFeats, totalMaps)

    print(mapFeats.shape)
    print(songFeats.shape)
    # mapFeats = sorted(mapFeats, key=lambda x: x[0])  # sort by id (makeshift title: song-mapper-diff)
    # songFeats = sorted(songFeats, key=lambda x: x[0])  # similar to above
    # no need to do above sorting with new implementation, they already line up.
    # sloppy but at this point we expect the IDs to basically line up. But notably there are usually more mapFeats than songFeats: Some maps share a song.
    # print(mapFeats[8][0][0])
    # print(songFeats[8][0][0])

    # for i in range(14):
    #     print(len(mapFeats[i]), len(songFeats[i]))
    # print(songFeats[0][0][1].shape)
    # createConvLSTM()

    # createModel(mapFeats, songFeats)
    # print(songFeats[0][1][82])
    # print(mapFeats.shape)
    # print()
    # print(songFeats[0][1].shape)

# basicModel()
createConvLSTM()