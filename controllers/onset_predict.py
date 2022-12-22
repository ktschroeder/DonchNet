import feature_extraction.audio_to_feats as atf
import numpy as np
from numpy import *
import pickle
# import keras.models
import config
import keras
from dataset_tools import audio_converter as ac
from dataset_tools import dataset_management as dm
import os

max_sequence_length = config.audioLengthMaxSeconds * 100  # 100 frames per second, or 10 ms per frame




def batchGetSongFeatsFromAudios(songFeatPaths):  # needs to also analyze audio instead of just loading feats data, which hasn't been computed for a new audio
    songFeats = []
    tempWavDir = "data/temp/temp.wav"
    for item in songFeatPaths:

        ext = os.path.splitext(item)[-1].lower()

        if ext == ".pkl":  # in this case we epect this is the pre-processed frequency info. This is convenient for onset metrics analysis
            file = open(item, 'rb')
            audioFeat = pickle.load(file)

        else:

            assert(ext == ".mp3" or ext == ".ogg")

            # this includes standardization
            if ext == ".mp3":
                ac.mp3ToWav(item, tempWavDir)
            if ext == ".ogg":
                ac.oggToWav(item, tempWavDir)
            audioFeat = atf.makeFeats(tempWavDir)
            os.remove(tempWavDir)

            bandMeans = np.loadtxt("data/misc/bandMeans.txt")
            bandStdevs = np.loadtxt("data/misc/bandStdevs.txt")

            audioFeat = dm.normalize(audioFeat, bandMeans, bandStdevs)

        songFeats.append([item[:-4], audioFeat])  
    return songFeats




def batchPrepareSongFeatsForModel(songFeats):
    mapCount = len(songFeats)
    pSongFeats = np.full((mapCount, max_sequence_length, 15, 40), config.pad)  # incoming as (mapCount, max_sequence_length,15,40)
    # starRatings = np.empty(mapCount, dtype=float32)
    
    for i in range(len(songFeats)):

        audioFrames = len(songFeats[i][1])
        # starRatings[i] = float32(SRs[i])  # star rating of map (difficulty)
            
        trimmedSongFeats = songFeats[i][1][:min(len(songFeats[i][1]), max_sequence_length)]  # trim to max sequence length if applicable
        pad = []
        for k in range (max_sequence_length - len(trimmedSongFeats)):
            # print("got in with", max_sequence_length - len(trimmedSongFeats))
            pad.append(np.full((15,40), config.pad))
        # print(pad)
        if(len(pad) > 0):
            trimmedSongFeats = np.vstack((trimmedSongFeats, pad))
        pSongFeats[i] = trimmedSongFeats

    return pSongFeats #, starRatings

class Custom_Prediction_Generator(keras.utils.Sequence): # via https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71

    def addContext(self, x, bookendLength):
    # x.reshape(())   (config.audioLengthMaxSeconds, 40)?
        padding = np.full((1,40), config.pad, dtype=float32)  # TODO after normalization, this should be like -3 instead of -500. track minimum to determine this
        context = bookendLength
        window = 2*context + 1 # prepend and append context
        # want to create input_shape=(max_sequence_length,15,40) from (max_sequence_length, 40)
        out = np.zeros((len(x),window,40))
        for i in range (len(x)):
            bookended = np.zeros((window,40), dtype=float32)
            for j in range (context*-1, context+1):
                indexToGet = i + j  # if at start of audio this is negative in first half, if at end this is out of bounds positive in second half
                if indexToGet < 0 or indexToGet >= len(x):
                    bookended[j + context] = padding
                else:
                    bookended[j + context] = x[i + j] 
            out[i] = bookended
        return out
    
    def __init__(self, songFilenames, SRs, batch_size) :
        self.songFilenames = songFilenames
        self.batch_size = batch_size
        self.SRs = SRs
    
    def __len__(self) :
        return (np.ceil(len(self.songFilenames) / float(self.batch_size))).astype(np.int)
  
    def __getitem__(self, idx) :
        max_sequence_length = config.audioLengthMaxSeconds * 100  # 100 frames per second, or 10 ms per frame
        # print(f"---------:{idx} {self.batch_size}")
        batch_x = self.songFilenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_SRs = self.SRs[idx * self.batch_size : (idx+1) * self.batch_size]

        # Processing this batch to prepare it for the model
        bookendLength = 7  # number of frames to prepend and append on each side of central frame. For n, we have total 2n+1 frames.
        songFeats = batchGetSongFeatsFromAudios(batch_x)
        for songFeat in songFeats:
            songFeat[1] = self.addContext(songFeat[1], bookendLength)

        x = batchPrepareSongFeatsForModel(songFeats) # TODO SRs? How to get them (it) here? Necessary?

        # print("in generator: ", x.shape, y.shape, starRatings.shape)  # currently (17, 24000, 15, 40) (17, 24000) (17,)
    
        x = reshape(x, (len(x), len(x[0]), len(x[0][0]), len(x[0][0][0]),1))  # 17, 24000, 15, 40, 1
        x = x.astype(float32)
        # starRatings = starRatings.reshape((len(starRatings), 1, 1))

        starRatings = batch_SRs

        # process star ratings so they can be prepended properly before LSTMs
        stars = np.empty((len(starRatings), max_sequence_length, 1), dtype=float32)
        for i in range(len(stars)):
            stars[i] = np.full((max_sequence_length, 1), starRatings[i])
        starRatings = stars.astype(float32)

        return [x, starRatings], None  # TODO Added this None to stop error: ValueError: Layer "custom_train_step" expects 2 input(s), but it received 1 input tensors. Inputs received: [<tf.Tensor 'IteratorGetNext:0' shape=(None, None, None, None, None) dtype=float32>] 
        # return np.array([
        #     resize(imread('/content/all_images/' + str(file_name)), (80, 80, 3))
        #        for file_name in batch_x])/255.0, np.array(batch_y)

def makePredictionFromAudio(model, audioFiles, SRs):
    # max_sequence_length = config.audioLengthMaxSeconds * 100  # 100 frames per second, or 10 ms per frame

    # audioFeats = atf.makeFeats(audioFile)  # assumes this is a WAV file already
    # pSongFeats = np.full((1, max_sequence_length, 15, 40), config.pad)  
    
    # # pSongFeats.append(tf.convert_to_tensor(songFeats[i][0][1], dtype=tf.int32))
    # trimmedSongFeats = audioFeats[:min(len(audioFeats), max_sequence_length)]  # trim to max sequence length if applicable
    # pad = []
    # for k in range (max_sequence_length - len(trimmedSongFeats)):
    #     # print("got in with", max_sequence_length - len(trimmedSongFeats))
    #     pad.append(np.full((15, 40), config.pad))
    # # print(pad)
    # if(len(pad) > 0):
    #     trimmedSongFeats = np.vstack((trimmedSongFeats, pad))
    # pSongFeats[0] = trimmedSongFeats

    # x = pSongFeats
    # x = reshape(x, (len(x), len(x[0]), len(x[0][0]), len(x[0][0][0]),1))  # 17, 30000, 15, 40, 1
    # x = x.astype(float32)

    # stars = np.empty((1, max_sequence_length, 1), dtype=float32)
    # for i in range(len(stars)):  # overkill for one prediction but scales
    #     stars[i] = np.full((max_sequence_length, 1), sr)


    #SRs = [5.0]  #TODO look here first if problem, consider shapes
    X_filenames = audioFiles
    generator_batch_size = 1  # don't confuse this with proportion of data used, like in training. Must be an int
    my_prediction_batch_generator = Custom_Prediction_Generator(X_filenames, SRs, generator_batch_size)

    prediction = model.predict(my_prediction_batch_generator, batch_size=generator_batch_size, verbose=1)

    # with open('models/prediction.pickle', 'wb') as handle:
    #     pickle.dump(prediction, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return prediction

def processPrediction(prediction):  # prediction here is the tensor of frequency bands created in makePredictionFromAudio.
    prediction = prediction[0] # assuming only one prediction, this reshapes data. If many predictions, implement for each element in input.
    print("Prediction sample:")
    print(prediction)
    return "placeholder"
