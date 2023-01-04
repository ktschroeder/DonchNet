import os
from controllers import onset_predict, onset_generate_taiko_map
import tensorflow as tf
from tensorflow import keras
import json
from feature_extraction import map_json_to_feats
from scipy.signal import argrelextrema
import numpy as np
from statistics import mean

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
    groundTruthOnsets = []
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

                mapOnsets = []
                for i in range (len(j["hitobjects"])):
                    mapOnsets.append(j["hitobjects"][i]["time"])  # offset in ms, currently ignoring type of object
                groundTruthOnsets.append(mapOnsets)
    # now we have all file names in mapFeatsFiles and songFeatsFiles

    precisions = []
    recalls = []
    fScores = []
    print(f"start... for {len(mapFeatsFiles)} maps...")
    # make predictions for each
    for i in range(len(mapFeatsFiles)):
        if i % 50 == 0:
            print(f"map number {i}...")

        prediction = onset_predict.makePredictionFromAudio(model, [songFeatsFiles[i]], [SRs[i]])
        prediction = np.reshape(prediction, (len(prediction[0])))
        predictions_smoothed = np.convolve(prediction, np.hamming(5), 'same')
        maxima = argrelextrema(predictions_smoothed, np.greater_equal, order=1)[0]



        fScore, precision, recall, threshold = tuneThreshold(prediction, maxima, groundTruthOnsets[i])
        precisions.append(precision)
        recalls.append(recall)
        fScores.append(fScore)

        print(f"Threshold {threshold} gave F-score {fScore}.")

        # if i > 30:
        #     break

    print(f"Averages: F-score {mean(fScores)}, precision {mean(precisions)}, recall {mean(recalls)} across {len(fScores)} items.")
    

    # within each map/prediction pair, compare list of onsets. Include relevant math. Can try tuning for thresholds and report this as a special metric.

    # could do binary search on threshold to quickly find best to, say, .01 precision.
    # could also do summary stats for each choice of flat threshold across all data, and legitimately pick the best performing threshold (thus best model inclusive of threshold).
    print("got to end of metrics thing")
    return

def tuneThreshold(prediction, maxima, groundTruths):
    thresholdPrecision = 0.01  # precision for threshold at which to stop trying to improve
    minThreshold = 0.01
    maxThreshold = 0.99
    startThreshold = 0.10
    threshold = startThreshold
    prevThreshold = -1.0
    bestFScore = -1
    bestPrecision = -1
    bestRecall = -1
    bestThreshold = -1
    # while True: TODO binary search here? Local minima possible though because of harmonic nature of f score?
    
    thresholds = [x / 100.0 for x in range(100)]  # 0.01, 0.02, ..., 0.99
    # while abs(threshold - prevThreshold) > thresholdPrecision:
    for threshold in thresholds:
        fScore, precision, recall = tuneThresholdHelper(prediction, maxima, groundTruths, threshold)
        if fScore > bestFScore:
            bestFScore = fScore
            bestRecall = recall
            bestPrecision = precision
            bestThreshold = threshold

    return bestFScore, bestPrecision, bestRecall, bestThreshold
    # return fScore, precision, recall, threshold

def tuneThresholdHelper(prediction, maxima, groundTruths, threshold):
    onsets = []
    for i in maxima:
        if prediction[i] >= threshold:
            t = float(i) * 10  # 10 ms per frame
            onsets.append(t)
    fScore, precision, recall = getFScore(onsets, groundTruths)
    return fScore, precision, recall


def getFScore(onsets, groundTruths):
    onsets = np.asarray(onsets)
    groundTruths = np.asarray(groundTruths)
    # print(onsets.shape)
    # print(groundTruths.shape)
    onsets = list(map(int, onsets))
    groundTruths = list(map(int, groundTruths))

    assert(onsets == sorted(onsets))
    assert(groundTruths == sorted(groundTruths))

    window = 25  # radius in milliseconds of window in which predicted onset will be considered a true positive
    shift = -5  # ms to shift window (positive shifts it later in time, negative earlier). -10 compensates for model's placement policy.
    truePositives = 0
    falsePositives = 0
    trueNegatives = 0
    falseNegatives = 0
    onsetIndex = 0
    truthIndex = 0
    prevPositiveGroundTruth = -1000  # time of ground truth for previous true positive
    duration = max(max(onsets, default = 0), max(groundTruths))  # duration in milliseconds
    outOfOnsets = 0
    outOfTruths = 0
    rounds = 0

    while True:
        rounds += 1
        if onsetIndex >= len(onsets):
            outOfOnsets = 1
        if truthIndex >= len(groundTruths):
            outOfTruths = 1

        if outOfOnsets and outOfTruths:
            break
        if outOfOnsets:
            falseNegatives += 1
            truthIndex += 1
            continue
        if outOfTruths:
            falsePositives += 1
            onsetIndex += 1
            continue

        onset = onsets[onsetIndex]
        truth = groundTruths[truthIndex]
        if onset < truth - window + shift:  # if onset is much earlier than next truth (then we must abandon that onset)
            falsePositives += 1
            onsetIndex += 1
        elif onset > truth + window + shift:  # if onset is much later than next truth (then we must abandon that truth)
            falseNegatives += 1
            truthIndex += 1
        else:  # then onset and truth are within the window distance
            truePositives += 1
            onsetIndex += 1
            truthIndex += 1


    # for frame in range(duration / 10):  # 10 ms frames
    #     time = frame*10
    #     nextGroundTruth = groundTruths[truthIndex]  # may throw out of bounds exception TODO
    #     if time in onsets and time > prevPositiveGroundTruth + window:  # positive
    #         if time >= nextGroundTruth - window and time <= nextGroundTruth + window:  # if onset is within the window around the groundTruth
    #             truePositives += 1
    #             prevPositiveGroundTruth = nextGroundTruth
    #             truthIndex += 1
    #         else:  # false positive
    #             falsePositives += 1

    #     else:  # negative
    #         if (time > prevPositiveGroundTruth + window) and (time >= nextGroundTruth - window and time <= nextGroundTruth + window):  # if onset is within the window around the groundTruth
    #             falseNegatives += 1
    #         else:
    #             trueNegatives += 1

    assert(truePositives + falsePositives == len(onsets))

    
    if truePositives == 0:
        precision = 0.00001
        recall = precision
    else:
        precision = truePositives / (truePositives + falsePositives)
        recall = truePositives / (truePositives + falseNegatives)

    # print(f"rounds {rounds}, onsets {len(onsets)}, truths {len(groundTruths)}, truePositives {truePositives}, falsePositives {falsePositives}, predicted onsets {len(onsets)}, precision {precision}, recall {recall}")
    fScore = 2 * (precision * recall) / (precision + recall)
    return fScore, precision, recall

onsetMetrics()