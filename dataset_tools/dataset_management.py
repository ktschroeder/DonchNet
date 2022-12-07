import sys
sys.path.append("C:/Users/Admin/Documents/GitHub/taiko_project")   # TODO cleaner way to do this?

import os
import map_to_json_converter
from feature_extraction import audio_to_feats as atf
# from dataset_tools import dataset_management as dm
from dataset_tools import audio_converter as ac
import traceback
import numpy as np
import pickle
import config

from misc_tools import slugify

def normalize(feats, bandMeans, bandStdevs):
    feats -= bandMeans
    feats /= bandStdevs
    return feats


def jointlyMakeJsonsAndMels():
    debug = 0  # number of maps to skip over, for debugging or to start at some point. Set this to 0 for normal use of this function.
    mainDir = config.datasetMainDirectory
    storageDir = config.featureMainDirectory
    tempWavDir = "data/temp/temp.wav"
    means = []  # tracked for audio feats to help with normalization
    stdevs = []  # tracked for audio feats to help with normalization
    #simpleMin = 99999  # for tracking minimum value seen in audio feats (useful for padding) ... Currently, this will necessarily be -500.
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/temp"):
        os.makedirs("data/temp")
    if not os.path.exists("data/misc"):
        os.makedirs("data/misc")
    if not os.path.exists(storageDir):
        os.makedirs(storageDir)
    if not os.path.exists("data/holdout_feats"):
        os.makedirs("data/holdout_feats")

    print(f"Analyzing songs folders in {mainDir}, which will be stored in {storageDir}...")

    processedAudios = 0
    for songFolder in os.listdir(mainDir):  # first make JSONs and get JSON info for ID that will be used in creating mels
        safeName = songFolder
        if len(songFolder) > 60:
            safeName = songFolder[:60] + "---"  # fixes issues with directory names in Windows, which have a length limit
        
        newDir = os.path.join(storageDir, safeName)
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

            audioFeats = atf.makeFeats(tempWavDir, newDir, songFolder)
            mean = np.mean(audioFeats, axis=0)  # returns the mean value at each frequency band
            stdev = np.std(audioFeats, axis=0)  # likewise with standard deviation
            means.append(mean)  # we are calcing across song's bands then across all songs. Could instead calc across all songs' bands. Slightly different effect. Insignificant?
            stdevs.append(stdev)

            os.remove(tempWavDir)
            processedAudios += 1
            if processedAudios < 99 and processedAudios % 10 == 0:
                print(f"Processed {processedAudios} audios...")
            elif processedAudios % 100 == 0:
                if processedAudios == 100:
                    print("From now on, only printing an update every 100 audios.")
                print(f"Processed {processedAudios} audios...")
            # if ext == ".wav":
            #     atf.makeFeats(os.path.join(mainDir, songFolder, item), newDir, songFolder)
    print(f"Finished making base feats. Processed {processedAudios} audios, pre-normalization. Normalizing...")
    bandMeans = np.mean(means, axis=0)
    bandStdevs = np.mean(stdevs, axis=0)
    # file = open("data/temp/bandMeans.txt", 'w')
    # file.write(bandMeans)
    # file = open("data/temp/bandStdevs.txt", 'w')
    # file.write(bandStdevs)
    # file.close()
    np.savetxt("data/misc/bandMeans.txt", bandMeans)
    np.savetxt("data/misc/bandStdevs.txt", bandStdevs)

    simpleMean = np.mean(bandMeans)
    simpleStdev = np.mean(bandStdevs)
    print(f"Overall simple mean and stdev are {simpleMean} and {simpleStdev} respectively...")

    normalized = 0

    for folder in os.listdir(storageDir):
        for file in os.listdir(os.path.join(storageDir, folder)):
            ext = os.path.splitext(file)[-1].lower()
            if not ext == ".pkl":
                continue
            with open(os.path.join(storageDir, folder, file), 'rb') as f:
                feats = pickle.load(f)
            feats = normalize(feats, bandMeans, bandStdevs)
            with open(os.path.join(storageDir, folder, file), 'wb') as f:
                pickle.dump(feats, f, protocol=pickle.HIGHEST_PROTOCOL)
            normalized += 1
            if normalized % 500 == 0:
                print(f"Normalized {normalized} audio feat sets...")

    
    print(f"Finished. Normalized {normalized} audio feat sets.")



jointlyMakeJsonsAndMels()