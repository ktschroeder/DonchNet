
#######################################################################################

import feature_extraction.audio_to_feats as atf
import numpy as np
from numpy import *
import pickle
# import keras.models
import config
import keras
from dataset_tools import audio_converter as ac
from dataset_tools import dataset_management as dm
import os
import feature_extraction.map_json_to_feats as jtf

max_sequence_length = config.audioLengthMaxSeconds * 100  # 100 frames per second, or 10 ms per frame




def batchGetSongFeatsFromAudios(songFeatPaths):  # needs to also analyze audio instead of just loading feats data, which hasn't been computed for a new audio
    songFeats = []
    tempWavDir = "data/temp/temp.wav"
    for item in songFeatPaths:

        ext = os.path.splitext(item)[-1].lower()
        assert(ext == ".mp3" or ext == ".ogg")

        # this includes standardization
        if ext == ".mp3":
            ac.mp3ToWav(item, tempWavDir)
        if ext == ".ogg":
            ac.oggToWav(item, tempWavDir)
        audioFeat = atf.makeFeats(tempWavDir)
        os.remove(tempWavDir)

        bandMeans = np.loadtxt("data/misc/bandMeans.txt")
        bandStdevs = np.loadtxt("data/misc/bandStdevs.txt")

        audioFeat = dm.normalize(audioFeat, bandMeans, bandStdevs)

        songFeats.append([item[:-4], audioFeat])  
    return songFeats




def batchPrepareFeatsForModel(mapInfo, songFeats, SRs):  # mapInfo has tuples of id, onsets, notes for each map. songFeats has 15x40 per onset. Need to make unrollings.

    onsetCount = 0
    for map in mapInfo:
        onsetCount += len(map[1])

    pMapFeats = np.empty((onsetCount, unrollings, 8), dtype=float32)  # Everything here should be filled TODO check
    pSongFeats = np.full((onsetCount, unrollings, 1+2*config.colorAudioBookendLength, 40), config.pad)
    pNotes = np.empty((onsetCount, 4), dtype=float32)
    
    onsetIndex = 0  
    # Stateless LSTM so should be fine to just mash all the unrollings together


    mapPad = np.array([0,0,0,0,0,0,0,0])
    audioPad = np.full((1+2*config.colorAudioBookendLength, 40), config.pad)
    for i, map in enumerate(mapInfo):
        song = songFeats[i]
        sr = map[3]

        onsets = np.array(map[1]).astype(int)

        rolledMapData = []
        # don: 0
        # kat: 2/8/10
        # fdon: 4
        # fkat: 6/12/14
        for j, color in enumerate(map[2]):
            color = int(color)
            don = 0
            kat = 0
            fdon = 0
            fkat = 0
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

            isFirstOnset = 0
            if j == 0:
                isFirstOnset = 1


            if j == 0:
                timeFromPrev = 0
                timeToNext = onsets[j+1] - onsets[j]
            elif j == len(onsets) - 1:
                timeFromPrev = onsets[j] - onsets[j-1]
                timeToNext = 0
            else:
                timeFromPrev = onsets[j] - onsets[j-1]
                timeToNext = onsets[j+1] - onsets[j]

            rolledMapData.append(np.array([don, kat, fdon, fkat, timeFromPrev / deltaTimeNormalizer, timeToNext / deltaTimeNormalizer, isFirstOnset, sr]))
            pNotes[j] = np.array([don, kat, fdon, fkat])


        # Now make unrollings from rolledMapData...
        for j, item in enumerate(rolledMapData):
            mapUnrollingsSet = np.empty((unrollings, 8), dtype=float32)
            songUnrollingsSet = np.empty((unrollings, 1+2*config.colorAudioBookendLength, 40), dtype=float32)
            for k in range(unrollings):  # 0 to 79
                indexToGet = j - unrollings + k  # first is 0 - 80 + 0, through 0 - 80 + 79
                if indexToGet < 0:
                    mapUnrollingsSet[k] = mapPad
                    songUnrollingsSet[k] = audioPad
                else:
                    assert(indexToGet < len(rolledMapData))
                    mapUnrollingsSet[k] = rolledMapData[indexToGet]
                    songUnrollingsSet[k] = song[1][indexToGet]

            pMapFeats[onsetIndex] = mapUnrollingsSet
            pSongFeats[onsetIndex] = songUnrollingsSet
            onsetIndex += 1

    assert(onsetIndex == onsetCount)     



    return pMapFeats, pSongFeats




class Custom_Prediction_Generator(keras.utils.Sequence): # via https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71

    def addContextAndTrim(self, x, onsets, bookendLength):
    # x.reshape(())   (config.audioLengthMaxSeconds, 40)?
        padding = np.full((1,40), config.pad, dtype=float32)  # TODO after normalization, this should be like -3 instead of -500. track minimum to determine this
        context = bookendLength
        window = 2*context + 1 # prepend and append context
        # want to create input_shape=(max_sequence_length,15,40) from (max_sequence_length, 40)
        out = np.zeros((len(onsets),window,40))
        onsets = np.array(onsets).astype(int)
        # #
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
    
    def __init__(self, mapFiles, audioFiles, SRs, generator_batch_size) :
        self.mapFilenames = mapFiles
        self.songFilenames = audioFiles
        self.batch_size = generator_batch_size
        self.SRs = SRs
    
    def __len__(self) :
        return (np.ceil(len(self.songFilenames) / float(self.batch_size))).astype(np.int)
  
    def __getitem__(self, idx) :
        max_sequence_length = config.audioLengthMaxSeconds * 100  # 100 frames per second, or 10 ms per frame  # TODO unless this is increased in prediction (could double it) longer songs may have odd behavior?
        # print(f"---------:{idx} {self.batch_size}")
        batch_maps = self.mapFilenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_songs = self.songFilenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_SRs = self.SRs[idx * self.batch_size : (idx+1) * self.batch_size]

        # get onsets first and related info, including color
        mapInfo = batchGetMapInfo(batch_maps)  # id, onsets, notes for each map. Does not include SR since we will take the user provided SR for prediction

        # Processing this batch to prepare it for the model
        bookendLength = config.colorAudioBookendLength  # number of frames to prepend and append on each side of central frame. For n, we have total 2n+1 frames.
        songFeats = batchGetSongFeatsFromAudios(batch_songs)

        for i, songFeat in enumerate(songFeats):  # TODO only do this for the feats matching with the onsets. Much faster
            songFeat[1] = self.addContextAndTrim(songFeat[1], mapInfo[i][1], bookendLength)

        # Untested but perhaps good to go
        xMaps, xAudios = batchPrepareFeatsForModel(mapInfo, songFeats, batch_SRs)

        xMaps = reshape(xMaps, (len(xMaps), config.colorUnrollings, 8)).astype(float32)  # want |onsets|, 80, 4+4, 1. 4+4 is the 4 colors and 4 other map feats.
        xAudios = reshape(xAudios, (len(xMaps), config.colorUnrollings, 1+2*config.colorAudioBookendLength, 40, 1)).astype(float32)  # want |onsets|, 80, 15, 40, 1: onsets, unrollings, 1+2bookends, freq bands, 1.
                
        return [xMaps, xAudios], None


def batchGetMapInfo(batch_maps):
    mapFeats = []
    for item in batch_maps:
        id, onsets, notes = jtf.jsonToFeatsColor(item)
        cap = min(len(onsets), config.colorOnsetMax)  # TODO limiting, can remove maybe for prediction
        onsets = onsets[:cap]
        notes = notes[:cap]
        mapFeats.append([id, onsets, notes])
    return mapFeats

def makePredictionFromMapAndAudio(model, mapFiles, audioFiles, SRs): # returns prediction from model. in: model, mapFiles, audioFiles, starRatings
    
    generator_batch_size = 1  # don't confuse this with proportion of data used, like in training. Must be an int
    my_prediction_batch_generator = Custom_Prediction_Generator(mapFiles, audioFiles, SRs, generator_batch_size)

    prediction = model.predict(my_prediction_batch_generator, batch_size=generator_batch_size, verbose=1)

    with open('models/prediction.pickle', 'wb') as handle:
        pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return prediction

# def processPrediction(prediction):  # prediction here is the tensor of frequency bands created in makePredictionFromAudio.
#     prediction = prediction[0] # assuming only one prediction, this reshapes data. If many predictions, implement for each element in input.
#     print("Prediction sample:")
#     print(prediction)
#     return "placeholder"
