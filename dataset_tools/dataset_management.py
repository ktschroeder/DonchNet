import sys
sys.path.append("C:/Users/Admin/Documents/GitHub/taiko_project")   # TODO cleaner way to do this?

import os
import map_to_json_converter
from feature_extraction import audio_to_feats as atf
from dataset_tools import dataset_management as dm

def makeJsons():
    made = 0
    mainDir = "C:/Users/Admin/Documents/small_taiko_dataset"  # TODO temporary
    for songFolder in os.listdir(mainDir):
        for item in os.listdir(os.path.join(mainDir, songFolder)):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".osu":
                made += map_to_json_converter.mapToJson(os.path.join(mainDir, songFolder, item))
    print(f"made {made} JSONs")

#makeJsons()

def makeMels():  # want to send title and mapper
    jsons = dm.makeJsons
    made = 0
    mainDir = "C:/Users/Admin/Documents/small_taiko_dataset"  # TODO temporary
    for songFolder in os.listdir(mainDir):
        for item in os.listdir(os.path.join(mainDir, songFolder)):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".wav":
                made += atf.getFeats(os.path.join(mainDir, songFolder, item), songFolder, jsons)
    print(f"made Mels for {made} song folders")

#makeMels()