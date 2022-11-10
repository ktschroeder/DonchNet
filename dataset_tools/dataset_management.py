import sys
sys.path.append("C:/Users/Admin/Documents/GitHub/taiko_project")   # TODO cleaner way to do this?

import os
import map_to_json_converter
from feature_extraction import audio_to_feats as atf
# from dataset_tools import dataset_management as dm
from dataset_tools import audio_converter as ac

from misc_tools import slugify

# marked for deletion TODO
# def makeJsons():
#     made = 0
#     mainDir = "C:/Users/Admin/Documents/medium_taiko_dataset"  # TODO temporary
#     for songFolder in os.listdir(mainDir):
#         for item in os.listdir(os.path.join(mainDir, songFolder)):
#             ext = os.path.splitext(item)[-1].lower()
#             if ext == ".osu":
#                 made += map_to_json_converter.mapToJson(os.path.join(mainDir, songFolder, item))
#     print(f"made {made} JSONs")

#makeJsons()

# marked for deletion TODO
# def makeMels():  # want to send title and mapper
#     jsons = makeJsons
#     made = 0
#     mainDir = "C:/Users/Admin/Documents/medium_taiko_dataset"  # TODO temporary
#     for songFolder in os.listdir(mainDir):
#         for item in os.listdir(os.path.join(mainDir, songFolder)):
#             ext = os.path.splitext(item)[-1].lower()
#             if ext == ".wav":
#                 made += atf.getFeats(os.path.join(mainDir, songFolder, item), songFolder, jsons)
#     print(f"made Mels for {made} song folders")

#makeMels()

def jointlyMakeJsonsAndMels():
    mainDir = "C:/Users/Admin/Documents/medium_taiko_dataset"  # TODO temporary
    tempWavDir = "data/temp/temp.wav"
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/temp"):
        os.makedirs("data/temp")
    if not os.path.exists("data/stored_feats"):
        os.makedirs("data/stored_feats")
    if not os.path.exists("data/holdout_feats"):
        os.makedirs("data/holdout_feats")

    processedAudios = 0
    for songFolder in os.listdir(mainDir):  # first make JSONs and get JSON info for ID that will be used in creating mels
        safeName = songFolder
        if len(songFolder) > 60:
            safeName = songFolder[:60] + "---"  # fixes issues with directory names in Windows
        
        newDir = os.path.join("data/stored_feats", safeName)
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

        for item in os.listdir(os.path.join(mainDir, songFolder)):  # to conserve storage, we temporarily convert to WAV to standardize, create audio feats, then delete the WAV
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".mp3":
                ac.mp3ToWav(os.path.join(mainDir, songFolder, item), tempWavDir)
                atf.makeFeats(tempWavDir, newDir, songFolder)
                os.remove(tempWavDir)
                processedAudios += 1
                if processedAudios % 20 == 0:
                    print(f"Processed {processedAudios} audios...")
            if ext == ".ogg":
                ac.oggToWav(os.path.join(mainDir, songFolder, item), tempWavDir)
                atf.makeFeats(tempWavDir, newDir, songFolder)
                os.remove(tempWavDir)
                processedAudios += 1
                if processedAudios % 20 == 0:
                    print(f"Processed {processedAudios} audios...")
            # if ext == ".wav":
            #     atf.makeFeats(os.path.join(mainDir, songFolder, item), newDir, songFolder)

            if processedAudios > 30:
                break
    print("Finished making feats")

jointlyMakeJsonsAndMels()