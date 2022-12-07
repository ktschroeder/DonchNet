import feature_extraction.audio_to_feats as atf
import numpy as np
from numpy import *
import pickle
# import keras.models
import config

def makePredictionFromAudio(model, audioFile, sr):
    max_sequence_length = config.audioLengthMaxSeconds * 100  # 100 frames per second, or 10 ms per frame

    audioFeats = atf.makeFeats(audioFile)  # assumes this is a WAV file already
    pSongFeats = np.full((1, max_sequence_length, 15, 40), -500)
    
    # pSongFeats.append(tf.convert_to_tensor(songFeats[i][0][1], dtype=tf.int32))
    trimmedSongFeats = audioFeats[:min(len(audioFeats), max_sequence_length)]  # trim to max sequence length if applicable
    pad = []
    for k in range (max_sequence_length - len(trimmedSongFeats)):
        # print("got in with", max_sequence_length - len(trimmedSongFeats))
        pad.append(np.full((15, 40), -500))
    # print(pad)
    if(len(pad) > 0):
        trimmedSongFeats = np.vstack((trimmedSongFeats, pad))
    pSongFeats[0] = trimmedSongFeats

    x = pSongFeats
    x = reshape(x, (len(x), len(x[0]), len(x[0][0]), len(x[0][0][0]),1))  # 17, 30000, 15, 40, 1
    x = x.astype(float32)

    stars = np.empty((1, max_sequence_length, 1), dtype=float32)
    for i in range(len(stars)):  # overkill for one prediction but scales
        stars[i] = np.full((max_sequence_length, 1), sr)

    prediction = model.predict([x, stars], verbose=1)

    with open('models/prediction.pickle', 'wb') as handle:
        pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return prediction

def processPrediction(prediction):  # prediction here is the tensor of frequency bands created in makePredictionFromAudio.
    prediction = prediction[0] # assuming only one prediction, this reshapes data. If many predictions, implement for each element in input.
    print("Prediction sample:")
    print(prediction)
    return "placeholder"
