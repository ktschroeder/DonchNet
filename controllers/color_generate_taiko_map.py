import numpy as np
import config
import os
from misc_tools import slugify
import shutil
from misc_tools import color_generate_osu_file
from scipy.signal import argrelextrema

def convertColorPredictionToMap(prediction, audioFile, mapfile, name):
    # expecting prediction in form: [[0,0,1,0],[0,1,0,0], ... ] where each quadruplet is the color of the respective onset
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/prediction_maps"):
        os.makedirs("data/prediction_maps")
    name = slugify.slugify(name)
    path = f"data/prediction_maps/{name}"
    if not os.path.exists(path):
        os.makedirs(path)

    shutil.copyfile(audioFile, f"{path}/{os.path.basename(audioFile)}")

    osuFile = color_generate_osu_file.generateOsuFile(prediction, mapfile, name)

    file = open(f"{path}/map.osu", "w")
    file.write(osuFile)

    return file
