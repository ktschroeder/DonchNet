
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
deltaTimeNormalizer = 200  # number that timeFromPrev and timeToNext are divided by to make them more suitable for learning. Arbitrary choice: very roughly average distance between 2 onsets
finishers = config.permitFinishers


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

    pMapFeats = np.empty((onsetCount, 6+finishers), dtype=float32)  # Everything here should be filled TODO check
    pSongFeats = np.full((onsetCount, 1+2*config.colorAudioBookendLength, 40), config.pad)
    
    onsetIndex = 0

    # mapPad = np.array([-1,-1,-1,-1,-1,-1,-1,-1])
    # audioPad = np.full((1+2*config.colorAudioBookendLength, 40), config.pad)
    for i, map in enumerate(mapInfo):
        song = songFeats[i]
        sr = SRs[i]

        onsets = np.array(map[1]).astype(int)

        rolledMapData = []
        # don: 0
        # kat: 2/8/10
        # fdon: 4
        # fkat: 6/12/14
        for j, color in enumerate(map[2]):

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
            if finishers:
                rolledMapData.append(np.array([-1, -1, -1, -1, timeFromPrev / deltaTimeNormalizer, timeToNext / deltaTimeNormalizer, isFirstOnset, sr]))  # since we are predicting, all colors start 0
            else:
                rolledMapData.append(np.array([-1, -1, timeFromPrev / deltaTimeNormalizer, timeToNext / deltaTimeNormalizer, isFirstOnset, sr]))  # since we are predicting, all colors start 0
            

        # # Now make unrollings from rolledMapData...
        # for j, item in enumerate(rolledMapData):
        #     mapUnrollingsSet = np.empty((unrollings, 8), dtype=float32)
        #     songUnrollingsSet = np.empty((unrollings, 1+2*config.colorAudioBookendLength, 40), dtype=float32)
        #     for k in range(unrollings):  # 0 to 79
        #         indexToGet = j - unrollings + k  # first is 0 - 80 + 0, through 0 - 80 + 79
        #         if indexToGet < 0:
        #             mapUnrollingsSet[k] = mapPad
        #             songUnrollingsSet[k] = audioPad
        #         else:
        #             assert(indexToGet < len(rolledMapData))
        #             mapUnrollingsSet[k] = rolledMapData[indexToGet]
        #             songUnrollingsSet[k] = song[1][indexToGet]

        #     pMapFeats[onsetIndex] = mapUnrollingsSet
        #     pSongFeats[onsetIndex] = songUnrollingsSet
        #     onsetIndex += 1

    # assert(onsetIndex == onsetCount)     



    return rolledMapData, song[1]






def addContextAndTrim(x, onsets, bookendLength):
# x.reshape(())   (config.audioLengthMaxSeconds, 40)?
    padding = np.full((1,40), config.pad, dtype=float32)  # TODO after normalization, this should be like -3 instead of -500. track minimum to determine this
    context = bookendLength
    window = 2*context + 1 # prepend and append context
    # want to create input_shape=(max_sequence_length,15,40) from (max_sequence_length, 40)
    out = np.zeros((len(onsets),window,40))
    onsets = np.array(onsets).astype(int)
    # onsets = onsets.astype(int)
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

from dataset_tools import map_to_json_converter
def batchGetMapInfo(batch_maps):
    mapFeats = []
    for item in batch_maps:
        json = map_to_json_converter.mapToJson(item)
        id, onsets, notes, _ = jtf.jsonToFeatsColor(json)  # also passes back SR but we ignore that SR here
        cap = min(len(onsets), config.colorOnsetMax)  # TODO limiting, can remove maybe for prediction
        onsets = onsets[:cap]
        notes = notes[:cap]
        mapFeats.append([id, onsets, notes])
    return mapFeats

mapPad = np.array([-1,-1,-1,-1,-1,-1,-1,-1])
if not finishers:
    mapPad = np.array([-1,-1,-1,-1,-1,-1])
audioPad = np.full((1+2*config.colorAudioBookendLength, 40), config.pad)



def predict(unrollings, model, xMaps, xAudios, temperature): # initially via https://towardsdatascience.com/time-series-forecasting-with-recurrent-neural-networks-74674e289816
    #prediction_list = xMaps #close_data[-unrollings:]
    # TODO construct unrollings here per-onset
    # predictedColors = np.empty((len(xMaps), 4))
    
    # for i, onsetUnrolling in enumerate(xMaps):
    #     xMap = onsetUnrolling
    #     xAudio = xAudios[i]

    #     out = model.predict() #TODO

    #     predictedColors[i] = out
    #     # I would then need to add this prediction to the next 64 unrollings

    includeAudioFeats = config.includeAudioFeatsInColorModel

    originalMap = xMaps
    originalAudio = xAudios
    leadingSequenceMap = np.full((unrollings, 6+finishers), mapPad)
    leadingSequenceAudio = np.full((unrollings, 1+2*config.colorAudioBookendLength, 40), audioPad)

    # print(leadingSequenceMap.dtype.name + leadingSequenceAudio.dtype.name) 

    assert(leadingSequenceMap.dtype.name == "int32")
    assert(leadingSequenceAudio.dtype.name == "float64")

    colorPredictions = []

    totals = [0,0,0,0]  # totals of colors predicted
    totalsRaw = [0.0,0.0,0.0,0.0]  # totals of raw prediction probabilities
    if not finishers:
        totals = [0,0]  # totals of colors predicted
        totalsRaw = [0.0,0.0]  # totals of raw prediction probabilities
    
    for i in range(len(xMaps)):  # TODO append map and sound data at top of loop, then update color once predicted
        # print(f"i: {i} - Shape of leadingSequenceMap: ")
        # print(leadingSequenceMap.shape)

        leadingSequenceMap = np.append(leadingSequenceMap, reshape(originalMap[i], (1, 6+finishers)), axis=0)
        if includeAudioFeats:
            leadingSequenceAudio = np.append(leadingSequenceAudio, reshape(originalAudio[i], (1, 1+2*config.colorAudioBookendLength, 40)), axis=0)

        xMap = leadingSequenceMap[-(unrollings+1):]  # x gets last 64 onsets
        if includeAudioFeats:
            xAudio = leadingSequenceAudio[-(unrollings+1):]  # x gets last 64 onsets

        # xMap[-1]

        # print(f"i: {i} - Shape of xMap: ")
        # print(xMap.shape)

        xMap = reshape(xMap, (1, unrollings+1, 6+finishers))  # model expects input in batch form, shapes get wonky if no extra first dimension
        if includeAudioFeats:
            xAudio = reshape(xAudio, (1, unrollings+1, 1+2*config.colorAudioBookendLength, 40))

        # print("Shape of xMap: ")
        # print(xMap.shape)
        # print("Shape of xAudio: ")
        # print(xAudio.shape)

        # x = x.reshape((1, unrollings, 1))

        if includeAudioFeats:
            out = model.predict([xMap, xAudio], verbose=0)#[0][0]  # if verbose, prints for every call to model.predict
        else:
            out = model.predict([xMap], verbose=0)

        # update leading sequences with new onset based on the predicted color combined with original data
        colorPrediction = solidifyColorPrediction(out[0], temperature)  # extra dimension in output, again because things are in batch form
        colorPredictions.append(colorPrediction)
        totals = [a + b for a, b in zip(totals, colorPrediction)]
        totalsRaw = [a + b for a, b in zip(totalsRaw, out[0])]

        if finishers:
            newOnsetMap = np.append(colorPrediction, [originalMap[i][4], originalMap[i][5], originalMap[i][6], originalMap[i][7]])
        else:
            assert(len(colorPrediction) == 2)
            newOnsetMap = np.append(colorPrediction, [originalMap[i][2], originalMap[i][3], originalMap[i][4], originalMap[i][5]])
        # newOnsetAudio = originalAudio[i]

        newOnsetMap = reshape(newOnsetMap, (1, 6+finishers))
        leadingSequenceMap[-1] = newOnsetMap  # update onset with predicted color



        # newOnsetAudio = reshape(newOnsetAudio, (1, 1+2*config.colorAudioBookendLength, 40))

        # leadingSequenceMap = np.append(leadingSequenceMap, newOnsetMap, axis=0)
        # leadingSequenceAudio = np.append(leadingSequenceAudio, newOnsetAudio, axis=0)

        if i > 0 and i % 250 == 0:
            print(f"Predicted {i} colors so far...")

    assert(len(colorPredictions) == len(xMaps))
    return colorPredictions, totals, totalsRaw


def solidifyColorPrediction(out, temperature):
    don = [1,0,0,0]
    kat = [0,1,0,0]
    fdon = [0,0,1,0]
    fkat = [0,0,0,1]
    objects = [don,kat,fdon,fkat]

    if not finishers:
        don = [1,0]
        kat = [0,1]
        objects = [don, kat]

    assert(len(out) == 2+finishers)
    colorIndex = sample(out, temperature)
    color = objects[colorIndex]
    
    return color


def sample(preds, temperature):  
    # temp is originally 1.0. Low = predictability, high = stochasticity
    # temp of 0 is equivalent to basic argmax. infinity is equivalent to uniform sampling.
    # tempo of 1 is sampling from original probabilities. 
    # decreasing from 1, Probabilities become shifted toward larger ones and away from smaller ones
    # increasing from 1, probabilities become more enarly equal
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def makePredictionFromMapAndAudio(model, mapFiles, audioFiles, SRs, temperature): # returns prediction from model. in: model, mapFiles, audioFiles, starRatings
    
    print(f"Predicting with finishers {finishers}, includeAudio {config.includeAudioFeatsInColorModel}, temperature {temperature}")

    # get onsets first and related info, including color
    mapInfo = batchGetMapInfo(mapFiles)  # id, onsets, notes for each map. Does not include SR since we will take the user provided SR for prediction
    unrollings = config.colorUnrollings

    # Processing this batch to prepare it for the model
    bookendLength = config.colorAudioBookendLength  # number of frames to prepend and append on each side of central frame. For n, we have total 2n+1 frames.
    songFeats = batchGetSongFeatsFromAudios(audioFiles)

    for i, songFeat in enumerate(songFeats):  # TODO only do this for the feats matching with the onsets. Much faster
        songFeat[1] = addContextAndTrim(songFeat[1], mapInfo[i][1], bookendLength)

    # Untested but perhaps good to go
    xMaps, xAudios = batchPrepareFeatsForModel(mapInfo, songFeats, SRs)  # # everything should be same as color training, except all notes are colorless and we provided SR

    prediction = None
    colors = None

    xMaps = reshape(xMaps, (len(xMaps), 6+finishers)).astype(float32)  # want |onsets|, 80, 4+4, 1. 4+4 is the 4 colors and 4 other map feats.
    xAudios = reshape(xAudios, (len(xMaps), 1+2*config.colorAudioBookendLength, 40, 1)).astype(float32)  # want |onsets|, 80, 15, 40, 1: onsets, unrollings, 1+2bookends, freq bands, 1.
          
    print("Beginning prediction...")

    assert(len(audioFiles) == 1)  # All the onsets are mashed together in one list, would need to separate them by song to support several songs
    for i in range(len(audioFiles)):
        prediction, totals, totalsRaw = predict(unrollings, model, xMaps, xAudios, temperature)
    # prediction = model.predict(my_prediction_batch_generator, batch_size=generator_batch_size, verbose=1)

    if finishers:
        print(f"Predicted colors totals [don, kat, fdon, fkat]: {totals}")
        print(f"Raw prediction probability totals [don, kat, fdon, fkat]: {totalsRaw}")
    else:
        print(f"Predicted colors totals [don, kat]: {totals}")
        print(f"Raw prediction probability totals [don, kat]: {totalsRaw}")

    # print(prediction)
    print("Got to end of prediction")

    with open('models/prediction.pickle', 'wb') as handle:
        pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return prediction

# def processPrediction(prediction):  # prediction here is the tensor of frequency bands created in makePredictionFromAudio.
#     prediction = prediction[0] # assuming only one prediction, this reshapes data. If many predictions, implement for each element in input.
#     print("Prediction sample:")
#     print(prediction)
#     return "placeholder"
