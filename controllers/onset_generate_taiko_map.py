import numpy as np
import config
import os
from misc_tools import slugify
import shutil
from misc_tools import generate_osu_file
from scipy.signal import argrelextrema

threshold = 0.06 # 0.06 # TODO standardize this

def processPrediction(prediction, threshold):
    predictions_smoothed = np.convolve(prediction, np.hamming(5), 'same')
    maxima = argrelextrema(predictions_smoothed, np.greater_equal, order=1)[0]
    # peak picking process is from Dance Dance Convolution
    # after hamming, among local maxima, assign a hit to original values above the given threshold

    # for i in range(len(prediction)):
    #     if prediction[i] >= threshold:
    #         prediction[i] = 1
    #     else:
    #         prediction[i] = 0

    onsets = []
    for i in maxima:
        if prediction[i] >= threshold:
            t = float(i) * 10  # 10 ms per frame
            onsets.append(t)

    return onsets

def createMap(onsets, audioFile, name, starRating):
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/prediction_maps"):
        os.makedirs("data/prediction_maps")
    name = slugify.slugify(name)
    path = f"data/prediction_maps/{name}"
    if not os.path.exists(path):
        os.makedirs(path)

    shutil.copyfile(audioFile, f"{path}/audio.wav")

    osuFile = generate_osu_file.generateOsuFile(onsets, name, starRating)

    file = open(f"{path}/map.osu", "w")
    file.write(osuFile)

    return file

def convertOnsetPredictionToMap(prediction, audioFile, name, starRating):
    prediction = np.reshape(prediction, (config.audioLengthMaxSeconds * 100))
    onsets = processPrediction(prediction, threshold)
    file = createMap(onsets, audioFile, name, starRating)

    return file