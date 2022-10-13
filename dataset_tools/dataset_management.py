import sys
sys.path.append("C:/Users/Admin/Documents/GitHub/taiko_project")   # TODO cleaner way to do this?

import os
import map_to_json_converter
from feature_extraction import audio_to_feats as atf
from dataset_tools import dataset_management as dm

from misc_tools import slugify

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

def jointlyMakeJsonsAndMels():
    mainDir = "C:/Users/Admin/Documents/small_taiko_dataset"  # TODO temporary
    for songFolder in os.listdir(mainDir):  # first makem JSONs and get JSON info for ID that will be used in creating mels
        newDir = os.path.join("data/feat_pkls", songFolder)
        if not os.path.exists(newDir):
            os.mkdir(newDir)
        json = ""
        for item in os.listdir(os.path.join(mainDir, songFolder)):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".osu":
                json = map_to_json_converter.mapToJson(os.path.join(mainDir, songFolder, item), newDir)
        # title = json["metadata"]["Title"].strip()
        # mapper = json["metadata"]["Creator"].strip()
        # title = slugify(title)
        # mapper = slugify(mapper)
        # id = title + '-' + mapper

        for item in os.listdir(os.path.join(mainDir, songFolder)):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".wav":
                atf.makeFeats(os.path.join(mainDir, songFolder, item), newDir, songFolder)

jointlyMakeJsonsAndMels()