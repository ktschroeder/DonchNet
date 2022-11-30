import sys
sys.path.append("C:/Users/Admin/Documents/GitHub/taiko_project")   # TODO cleaner way to do this?

import os
import map_to_json_converter
from feature_extraction import audio_to_feats as atf
# from dataset_tools import dataset_management as dm
from dataset_tools import audio_converter as ac
import traceback

from misc_tools import slugify

def jointlyMakeJsonsAndMels():
    debug = 0  # number of maps to skip over, for debugging or to start at some point. Set this to 0 for normal use of this function.
    mainDir = "C:/Users/Admin/Documents/adhoc_thing"  # TODO temporary
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
                try:
                    json = map_to_json_converter.mapToJson(os.path.join(mainDir, songFolder, item), newDir)
                except:
                    print(f"mapToJson failed for {os.path.join(mainDir, songFolder, item)}, stack trace follows:")
                    traceback.print_exc()
                    assert(None)
        # title = json["metadata"]["Title"].strip()
        # mapper = json["metadata"]["Creator"].strip()
        # title = slugify(title)
        # mapper = slugify(mapper)
        # id = title + '-' + mapper

        for item in os.listdir(os.path.join(mainDir, songFolder)):  # to conserve storage, we temporarily convert to WAV to standardize, create audio feats, then delete the WAV
            
            ext = os.path.splitext(item)[-1].lower()
            if not (ext == ".mp3" or ext == ".ogg"):
                continue

            if(processedAudios < debug):
                processedAudios += 1
                if processedAudios == debug - 1:
                    print("Finished debug skip...")
                continue

            if ext == ".mp3":
                ac.mp3ToWav(os.path.join(mainDir, songFolder, item), tempWavDir)
            if ext == ".ogg":
                ac.oggToWav(os.path.join(mainDir, songFolder, item), tempWavDir)

            atf.makeFeats(tempWavDir, newDir, songFolder)
            os.remove(tempWavDir)
            processedAudios += 1
            if processedAudios < 99 and processedAudios % 10 == 0:
                print(f"Processed {processedAudios} audios...")
            elif processedAudios % 10 == 0:
                if processedAudios == 100:
                    print("From now on, only printing an update every 100 audios.")
                print(f"Processed {processedAudios} audios...")
            # if ext == ".wav":
            #     atf.makeFeats(os.path.join(mainDir, songFolder, item), newDir, songFolder)
    print(f"Finished making feats. Processed {processedAudios} audios.")

jointlyMakeJsonsAndMels()