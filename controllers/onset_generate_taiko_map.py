import numpy as np
import config
import os
from misc_tools import slugify
import shutil
from misc_tools import generate_osu_file

threshold = 0.05 # TODO standardize this
def processPrediction(prediction, threshold):
    for i in range(len(prediction)):
        if prediction[i] >= threshold:
            prediction[i] = 1
        else:
            prediction[i] = 0
    return prediction

def createMap(binaryPrediction, audioFile, name, starRating):
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/prediction_maps"):
        os.makedirs("data/prediction_maps")
    name = slugify.slugify(name)
    path = f"data/prediction_maps/{name}"
    if not os.path.exists(path):
        os.makedirs(path)

    shutil.copyfile(audioFile, f"{path}/audio.wav")

    osuFile = generate_osu_file.generateOsuFile(binaryPrediction, name, starRating)

    file = open(f"{path}/map.osu", "w")
    file.write(osuFile)

    return file# TODO

def convertOnsetPredictionToMap(prediction, audioFile, name, starRating):
    prediction = np.reshape(prediction, (config.audioLengthMaxSeconds * 100))
    # TODO hamming and stuff
    binaryPrediction = processPrediction(prediction, threshold)
    file = createMap(binaryPrediction, audioFile, name, starRating)

    return file