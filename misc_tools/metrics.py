import os
from controllers import onset_predict, onset_generate_taiko_map
import tensorflow as tf
from tensorflow import keras
import json
from feature_extraction import map_json_to_feats

def onsetMetrics():
    # import model
    model = tf.keras.models.load_model("models/onset")



    # audioFiles = [os.path.join(path, "1158131 Kudou Chitose - Nilgiri", "audio.mp3")]
    # name = "nilgiri"
    # starRatings = [3.41]

    # onsetThresholds = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.22, 0.25, 0.30, 0.35]  # required "confidence" for a prediction peak to be considered an onset
    # prediction = onset_predict.makePredictionFromAudio(model, audioFiles, starRatings)
    # processedPrediction = onset_predict.processPrediction(prediction) #TODO Presumably this will throw exceptions for more than one song
    # for h in range(len(audioFiles)):
    #     for i in onsetThresholds:
    #         th = "{0:.2f}".format(i)
    #         newName = name + f" - T{th}"  # append threshold to name
    #         onset_generate_taiko_map.convertOnsetPredictionToMap(prediction, audioFiles[h], newName, starRatings[h], i)


    # import holdout/test data
    # import map data (ground truths)
    mainDir = "data/holdout_feats"
    mapFeatsFiles = []
    songFeatsFiles = []
    SRs = []
    for dir in os.listdir(mainDir):
        path = os.path.join(mainDir, dir)
        #  We will need to virtually duplicate the audio feat file for each map feat file to be 1-to-1
        audioFeatFile = None
        for item in os.listdir(path):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".pkl":
                audioFeatFile = item
        if not audioFeatFile:
            print(f"Couldn't get audioFeatFile from {path}")
            assert(None)
            
        for item in os.listdir(path):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".json":  # and random.random() < dataProportion:  # if random float from 0 to 1 is greater than dataProportion, we skip this map
                songFeatsFiles.append(os.path.join(mainDir, dir, audioFeatFile))

                mapFile = os.path.join(mainDir, dir, item)
                mapFeatsFiles.append(mapFile)
                data = open(mapFile,'rt').read()
                j = json.loads(data)
                sr = j["sr"]
                SRs.append(sr)
    # now we have all file names in mapFeatsFiles and songFeatsFiles


    # make predictions for each
    for i in range(len(mapFeatsFiles)):
        prediction = onset_predict.makePredictionFromAudio(model, [songFeatsFiles[i]], [SRs[i]])
        print(prediction)

        if i > 4:
            break
    

    # within each map/prediction pair, compare list of onsets. Include relevant math. Can try tuning for thresholds and report this as a special metric.

    # could do binary search on threshold to quickly find best to, say, .01 precision.
    # could also do summary stats for each choice of flat threshold across all data, and legitimately pick the best performing threshold (thus best model inclusive of threshold).
    print("got to end of metrics thing")
    return

onsetMetrics()