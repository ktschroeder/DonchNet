# For each onset, one of: don, kat, finisher don, finisher kat, slider(start), finisher slider(start), spinner(start)
# Possibly restrict to dons and kats (and sliders and spinners?). Lengths of sliders/spinners may be infeasible to represent/predict.
# Maybe have "finisher" as a sepearate boolean
# 
# four one-hot outputs: don, kat, don finisher, kat finisher. Maybe even just dons/kats.
# 
# predict next token given the previous sequence of tokens (Sequence generation)
# a start token denoting first onset in map
# LSTM with 2 layers each 128 cells
# add musical context in the form of rhythmic features: 
#   time since previous step, time till next step
# Add audio feats? Worth considering. Gets too complex? taikonation suggests this as an idea
# minimize cross-entropy (for onsets, binary cross-entropy)
# 50% dropout like onset
# 64 steps of unrolling (64 onsets)(?)
# include SR? DDc did this but had overfitting. may work with larger dataset
# likewise DDC tried including audio features but overfitted. Said may work with larger dataset.
# 
# X: onsets of maps. Y: Ordered hit-object types (1-to-1 with onsets). Does this require padding? Hopefully not.
# Also part of X: For each onset, timeToPrev and timeToNext
# hard-compute context for each onset like in Wonderland example?
# 
# Note: memory needed for tensors in this subproblem will be much much smaller. Can probably use simple batching. But not if adding audio context. ==> maybe still, can get only frames matching onsets
# 
# Complication with audio context: only provide audio frames around the relevant onsets? No clear way to do it otherwise.
# 
# feature indicating "isFirstOnset"
# 

import os
# from ..feature_extraction import map_json_to_feats as jtf
import feature_extraction.map_json_to_feats as jtf
import feature_extraction.audio_to_feats as atf
import controllers.onset_predict, controllers.onset_generate_taiko_map
import pickle
import numpy as np
from numpy import *
import tensorflow as tf
from tensorflow import keras
# from keras import layers, models
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Conv2D, Dense, Flatten, TimeDistributed, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from tensorflow.keras.activations import tanh, elu, relu
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Optimizer

from controllers import color_predict

import config

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# TODO if issues in development, remove this
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # intended to suppress warning about noncallable functions when saving the model

mainDir = config.featureMainDirectory
unrollings = config.colorUnrollings  # 80
deltaTimeNormalizer = 200  # number that timeFromPrev and timeToNext are divided by to make them more suitable for learning. Arbitrary choice: very roughly average distance between 2 onsets

# GENERATOR via https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
def generatorPrep(dataProportion):
    filenames_counter = 0
    labels_counter = -1

    mapFeatsFiles = []
    songFeatsFiles = []
    index = 0
    mapCount = 0
    for dir in os.listdir(mainDir):
        # mapFeatsFiles.append([])
        # songFeatsFiles.append([])
        path = os.path.join(mainDir, dir)
        #  We will need to virtually duplicate the audio feat file for each map feat file to be 1-to-1
        audioFeatFile = None
        for item in os.listdir(path):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".pkl":
                audioFeatFile = item
        if not audioFeatFile:
            print(f"Couldn't get audioFeatFile from {path}")
            assert(None)
            
        for item in os.listdir(path):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".json" and random.random() < dataProportion:  # if random float from 0 to 1 is greater than dataProportion, we skip this map
                mapFeatsFiles.append(os.path.join(mainDir, dir, item))
                songFeatsFiles.append(os.path.join(mainDir, dir, audioFeatFile))
    # now we have all file names in mapFeatsFiles and songFeatsFiles

    songsShuffled, mapsShuffled = shuffle(songFeatsFiles, mapFeatsFiles)
    X_train_filenames, X_val_filenames, y_train_filenames, y_val_filenames = train_test_split(songsShuffled, mapsShuffled, test_size=0.2, random_state=1)

    return X_train_filenames, X_val_filenames, y_train_filenames, y_val_filenames


class My_Custom_Generator(keras.utils.Sequence): # via https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71

    def addContextAndTrim(self, x, onsets, bookendLength):
    # x.reshape(())   (config.audioLengthMaxSeconds, 40)?
        padding = np.full((1,40), config.pad, dtype=float32)  # TODO after normalization, this should be like -3 instead of -500. track minimum to determine this
        context = bookendLength
        window = 2*context + 1 # prepend and append context
        # want to create input_shape=(max_sequence_length,15,40) from (max_sequence_length, 40)
        out = np.zeros((len(onsets),window,40))
        onsets = np.array(onsets).astype(int)
        # #
        # hitIndex = 0
        # audioFrames = len(x)
        # groundTruths = []  # if -1, no ground truth. Else index of ground truth in audio feats.
        # groundTruth = 0
        # for j in range(audioFrames * 10):  # for each millisecond
        #     # if j / 10 >= unrollings:
        #     #     break
        #     if hitIndex < len(onsets) and int(onsets[hitIndex]) <= j:  # if there is a hit
        #         hitIndex += 1
        #         groundTruths.append(j // 10)
        # #         groundTruth = j // 10
        # #     if j % 10 == 9:  # at every 10 milliseconds we summarize the 10ms frame and reset
        # #         if groundTruth == -1:
        # #             continue
        # #         groundTruths.append(groundTruth)
        # #         # temp += groundTruth
        # #         groundTruth = -1
        # # # for j in range(len(groundTruths), unrollings):
        # # #     groundTruths.append(-1)
        # # # pMapFeats.append(tf.con
        # # #
        truths = onsets // 10

        for i, truth in enumerate(truths): # want: for each onset in onsets (need its corresponding index in x). Could scan and continue for misses.
            bookended = np.zeros((window,40), dtype=float32)
            for j in range (context*-1, context+1):
                indexToGet = int(truth) + j  # if at start of audio this is negative in first half, if at end this is out of bounds positive in second half
                if indexToGet < 0 or indexToGet >= len(x):
                    bookended[j + context] = padding
                else:
                    bookended[j + context] = x[indexToGet]
            out[i] = bookended
        return out
    
    def __init__(self, songFilenames, mapFilenames, batch_size, hesitancy) :
        self.songFilenames = songFilenames
        self.mapFilenames = mapFilenames
        self.batch_size = batch_size
        self.hesitancy = hesitancy
    
    def __len__(self) :
        return (np.ceil(len(self.songFilenames) / float(self.batch_size))).astype(np.int)
  
    def __getitem__(self, idx) :
        batch_songs = self.songFilenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_maps = self.mapFilenames[idx * self.batch_size : (idx+1) * self.batch_size]

        # get onsets first and related info, including color
        mapInfo = batchGetMapInfo(batch_maps)  # id, onsets, notes, sr for each map

        # each frame of audiofeats corresponds to 10 ms. Here we want audio frames only containin an offset... when bookending.

        # Processing this batch to prepare it for the model
        bookendLength = config.colorAudioBookendLength  # number of frames to prepend and append on each side of central frame. For n, we have total 2n+1 frames.
        songFeats = batchGetSongFeats(batch_songs)
        assert(len(songFeats) == len(mapInfo))

        for i, songFeat in enumerate(songFeats):  # TODO only do this for the feats matching with the onsets. Much faster
            songFeat[1] = self.addContextAndTrim(songFeat[1], mapInfo[i][1], bookendLength)

        # Now we have mapinfo (id, onsets, notes, sr) and songFeats corresponding to onsets

        


        # Untested but perhaps good to go
        xMaps, xAudios, yNotes = batchPrepareFeatsForModel(mapInfo, songFeats, self.hesitancy)

        xMaps = reshape(xMaps, (len(xMaps), config.colorUnrollings+1, 8)).astype(float32)  # want |onsets|, 80, 4+4, 1. 4+4 is the 4 colors and 4 other map feats.
        xAudios = reshape(xAudios, (len(xMaps), config.colorUnrollings+1, 1+2*config.colorAudioBookendLength, 40, 1)).astype(float32)  # want |onsets|, 80, 15, 40, 1: onsets, unrollings, 1+2bookends, freq bands, 1.
        yNotes = reshape(yNotes, (len(xMaps), 4)).astype(float32)  # want |onsets|, 4. 4 comes from the one-hot for don, kat, fdon, fkat.
        

        # print("in generator: ", x.shape, y.shape, starRatings.shape)  # currently (17, 24000, 15, 40) (17, 24000) (17,)
    
        # x = reshape(x, (len(x), len(x[0]), len(x[0][0]), len(x[0][0][0]),1))           #  old: 17, 24000, 15, 40, 1
        # y = y.reshape((len(y),len(y[0]),1,1))
        # x = x.astype(float32)
        # starRatings = starRatings.reshape((len(starRatings), 1, 1))

        # process star ratings so they can be prepended properly before LSTMs
        # stars = np.empty((len(starRatings), max_sequence_length, 1), dtype=float32)
        # for i in range(len(stars)):
        #     stars[i] = np.full((max_sequence_length, 1), starRatings[i])
        # starRatings = stars

        # return [x, starRatings], y

        
        return [xMaps, xAudios], yNotes
        # return np.array([
        #     resize(imread('/content/all_images/' + str(file_name)), (80, 80, 3))
        #        for file_name in batch_x])/255.0, np.array(batch_y)


def batchGetMapInfo(batch_maps):
    mapFeats = []
    for item in batch_maps:
        id, onsets, notes, sr = jtf.jsonToFeatsColor(item)
        cap = min(len(onsets), config.colorOnsetMax)
        onsets = onsets[:cap]
        notes = notes[:cap]
        mapFeats.append([id, onsets, notes, sr])
    return mapFeats

def batchGetSongFeats(songFeatPaths):
    songFeats = []
    for item in songFeatPaths:
        file = open(item, 'rb')
        data = pickle.load(file)
        cap = min(len(data), config.colorOnsetMax)
        data = data[:cap]
        songFeats.append([item[:-4], data])  # the -4 removes ".pkl"  # TODO as it stands, this includes the full path. Probably don't want this.
    return songFeats

def batchGetMapFeats(mapFeatPaths):
    mapFeats = []
    for item in mapFeatPaths:
        id, onsets, sr = jtf.jsonToFeats(item)
        mapFeats.append([id, onsets, sr])
    return mapFeats

def batchPrepareFeatsForModel(mapInfo, songFeats, hesitancy):  # mapInfo has tuples of id, onsets, notes, sr for each map. songFeats has 15x40 per onset. Need to make unrollings.

    onsetCount = 0
    for map in mapInfo:
        onsetCount += len(map[1])

    hesitant = np.empty((onsetCount))
    goAheads = 0
    for i in range(len(hesitant)):
        h = 1
        if random.random() > hesitancy:  # if random float from 0 to 1 is over hesitancy then we include the associated onset, else skip
            h = 0
            goAheads += 1
        hesitant[i] = h
    if goAheads == 0:  # for stability we force at least 1 onset to be considered
        hesitant[random.randrange(onsetCount)] = 0
        goAheads = 1

    pMapFeats = np.empty((goAheads, unrollings+1, 8), dtype=float32)  # Everything here should be filled TODO check
    pSongFeats = np.full((goAheads, unrollings+1, 1+2*config.colorAudioBookendLength, 40), config.pad)
    pNotes = np.empty((goAheads, 4), dtype=float32)
    
    onsetIndex = 0  
    # Stateless LSTM so should be fine to just mash all the unrollings together


    mapPad = np.array([-1,-1,-1,-1,-1,-1,-1,-1])
    audioPad = np.full((1+2*config.colorAudioBookendLength, 40), config.pad)
    colorIndex = -1  # hack for hesitancy
    pNotesIndex = 0
    for i, map in enumerate(mapInfo):
        song = songFeats[i]
        sr = map[3]

        onsets = np.array(map[1]).astype(int)

        rolledMapData = []
        # don: 0
        # kat: 2/8/10
        # fdon: 4
        # fkat: 6/12/14
        # onsetIndexHesitancyHack = -1
        for j, color in enumerate(map[2]):
            colorIndex += 1
            # onsetIndexHesitancyHack += 1
            # if hesitant[onsetIndex + onsetIndexHesitancyHack] == 1:
                # continue

            color = int(color)
            don = -1
            kat = -1
            fdon = -1
            fkat = -1
            if color == 0:
                don = 1
            elif color == 2 or color == 8 or color == 10:
                kat = 1
            elif color == 4:
                fdon = 1
            elif color == 6 or color == 12 or color == 14:
                fkat = 1
            else:
                print(f"Found unexpected color (hitsound) {color} in map {map[0]}.")
                assert(None)
            assert(don or kat or fdon or fkat)

            isFirstOnset = -1
            if j == 0:
                isFirstOnset = 1


            if j == 0:
                timeFromPrev = -1
                timeToNext = onsets[j+1] - onsets[j]
            elif j == len(onsets) - 1:
                timeFromPrev = onsets[j] - onsets[j-1]
                timeToNext = -1
            else:
                timeFromPrev = onsets[j] - onsets[j-1]
                timeToNext = onsets[j+1] - onsets[j]

            rolledMapData.append(np.array([don, kat, fdon, fkat, timeFromPrev / deltaTimeNormalizer, timeToNext / deltaTimeNormalizer, isFirstOnset, sr]))
            
            if hesitant[colorIndex] == 0:  # if we are taking this color
                c = [don, kat, fdon, fkat]
                for i in range(4):
                    if c[i] == -1:
                        c[i] = 0
                pNotes[pNotesIndex] = np.array(c)
                pNotesIndex += 1


        # Now make unrollings from rolledMapData...
        for j, item in enumerate(rolledMapData):
            if hesitant[j] == 1:
                continue

            mapUnrollingsSet = np.empty((unrollings+1, 8), dtype=float32)
            songUnrollingsSet = np.empty((unrollings+1, 1+2*config.colorAudioBookendLength, 40), dtype=float32)
            for k in range(unrollings+1):  # 0 to 79
                indexToGet = j - (unrollings+1) + k  # first is 0 - 80 + 0, through 0 - 80 + 79
                if indexToGet < 0:
                    mapUnrollingsSet[k] = mapPad
                    songUnrollingsSet[k] = audioPad
                elif indexToGet == unrollings:  # this is the onset we would like to predict a color for
                    mapData = rolledMapData[indexToGet]
                    assert(len(mapData) == 8)
                    uncoloredMapData = np.append([-1,-1,-1,-1], mapData[4:])  # combine colorless with last 4 elements of genuine map data
                    mapUnrollingsSet[k] = uncoloredMapData
                    songUnrollingsSet[k] = song[1][indexToGet]
                else:
                    assert(indexToGet < len(rolledMapData))
                    mapUnrollingsSet[k] = rolledMapData[indexToGet]
                    songUnrollingsSet[k] = song[1][indexToGet]

            pMapFeats[onsetIndex] = mapUnrollingsSet
            pSongFeats[onsetIndex] = songUnrollingsSet
            onsetIndex += 1

    assert(onsetIndex == goAheads)      

    return pMapFeats, pSongFeats, pNotes  # previously pMapFeats, pSongFeats, starRatings

def initializer():
    return tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

def createColorModel():

    ######################################################################################################
    #
    #
    dataProportion = 0.02  # estimated portion (0 to 1) of data to be used. Based on randomness, so this is an estimate, unless it's 1.0, which uses all data.
    epochs = 50
    hesitancy = 0.005  # probabaility each onset will be considered. The idea is to take only a few onsets from each map, to prevent overfitting. 1000 onsets in a map * 0.01 ==> ~10 onsets
                      # in all cases, at least one onset is taken from each map considered (for stability)

    batch_size = 1  # as it stands, color model updates after every map (which consists of many onsets, granted. But they are all from same song.)
    learning_rate = 0.00001  # was 0.01 originally. 0.00001 also an option.
    hidden_units_lstm = 128 #128

    generator_batch_size = 1  # TODO pick near as large as possible for speed? This results in trying to allocate the tensor in memory for some reason. 3 is OOM for onset.
    #
    #
    ######################################################################################################

    X_train_filenames, X_val_filenames, y_train_filenames, y_val_filenames = generatorPrep(dataProportion) #TODO

    print(f"Color training will use {len(X_train_filenames)} maps and validation will use {len(X_val_filenames)} maps, via dataProportion {dataProportion}.")
    
    
    my_training_batch_generator = My_Custom_Generator(X_train_filenames, y_train_filenames, generator_batch_size, hesitancy)
    my_validation_batch_generator = My_Custom_Generator(X_val_filenames, y_val_filenames, generator_batch_size, hesitancy)  # inappropriate to use randomness in validation?


    clear_session()

    mainInput = tf.keras.Input(shape=(unrollings+1,8))  # 8: don, fdon, kat, fkat, timeFromPrev, timeToNext, isFirst, SR (TODO divide SR by 10 to normalize some)
    audioInput = tf.keras.Input(shape=(unrollings+1,1+2*config.colorAudioBookendLength,40,1))
    

    # base_maps = tf.keras.layers.Lambda(context)(input)
    base_maps = TimeDistributed(Conv2D(10, (3,3),activation='relu', padding='same',data_format='channels_last',kernel_initializer=initializer()))(audioInput)  # TODO could get these pre-trained via onset model? Maybe not worth it
    base_maps = TimeDistributed(MaxPool2D(pool_size=(1,3), padding='same'))(base_maps) # TODO is pooling correct with respect to dimensions?
    base_maps = TimeDistributed(Conv2D(20, (3,3),activation='relu', padding='same',data_format='channels_last',kernel_initializer=initializer()))(base_maps)
    base_maps = TimeDistributed(MaxPool2D(pool_size=(1,3), padding='same'))(base_maps)
    base_maps = TimeDistributed(Flatten())(base_maps) # see above notes, does this overly flatten temporal?

    merged = tf.keras.layers.Concatenate()([mainInput, base_maps])

    base_maps = LSTM(hidden_units_lstm, return_sequences=True,kernel_initializer=initializer())(merged)
    base_maps = Dropout(0.5, noise_shape=(None,1,hidden_units_lstm))(base_maps)  
    base_maps = LSTM(hidden_units_lstm, return_sequences=False,kernel_initializer=initializer())(base_maps)  
    base_maps = Dropout(0.5)(base_maps)   # TODO noise shape may be incorrect now *************************************** noise_shape=(None,1,hidden_units_lstm
    # base_maps = Dense(256, activation='relu')(base_maps)
    # base_maps = Dropout(0.5)(base_maps)
    # base_maps = Dense(128, activation='relu')(base_maps)
    # base_maps = Dropout(0.5)(base_maps) 

    # base_maps = Dense(32, activation='relu',kernel_initializer=initializer())(base_maps)
    # base_maps = Dropout(0.5)(base_maps) 

    base_maps = Dense(4, activation='softmax',kernel_initializer=initializer())(base_maps)

    color_model = keras.Model(inputs=[mainInput, audioInput], outputs=[base_maps])

    # bind all
    color_model.compile(  #ga for gradient accumulation
        loss = 'categorical_crossentropy',  # TODO may be source of overfitting if new model won't improve beyond 0.48 accuracy
        # metrics = ['accuracy'],
        optimizer = tf.keras.optimizers.SGD(momentum=0.9, nesterov=True, learning_rate=learning_rate), #TODO consider others?  # optimizer=keras.optimizers.Adam(1e-3),
        metrics=["accuracy"])

    checkpoint_filepath = 'models/checkpointColor'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='accuracy', 
        mode='max',
        save_best_only=True)

    # history = ga_model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, )
    history = color_model.fit(my_training_batch_generator, validation_data=my_validation_batch_generator, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[model_checkpoint_callback])
    print(color_model.summary())


    color_model.save("models/color")
    

    with open('models/history.pickle', 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)




createColorModel()

# print("Training finished...")

##################

# model = tf.keras.models.load_model("models/color")

# audioFiles = ["sample_onset_maps/urushi_t008/audio.mp3"]
# mapFiles = ["sample_onset_maps/urushi_t008/map.osu"]
# name = "Urushi"  # TODO need to update this if used for more than one song
# starRatings = [5.0]
# assert(len(audioFiles) == len(starRatings) and len(audioFiles) == len(mapFiles))  # cardinalities of these must be equal (and in respective order), they match 1-to-1 in the model

# prediction = controllers.color_predict.makePredictionFromMapAndAudio(model, mapFiles, audioFiles, starRatings)

##################


# print(prediction)
# for h in range(len(audioFiles)):
#     controllers.color_generate_taiko_map.convertOnsetPredictionToMap(prediction, mapfiles[h], audioFiles[h], name, starRatings[h])







print("Got to end of color model controller")