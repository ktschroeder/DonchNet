import os
# from ..feature_extraction import map_json_to_feats as jtf
import feature_extraction.map_json_to_feats as jtf
import feature_extraction.audio_to_feats as atf
import controllers.onset_predict, controllers.onset_generate_taiko_map
import pickle
import numpy as np
from numpy import *
import tensorflow as tf
from tensorflow import keras
# from keras import layers, models
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Conv2D, Dense, Flatten, TimeDistributed, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from tensorflow.keras.activations import tanh, elu, relu
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Optimizer

import config


# import runai.ga.keras

# sess_cpu = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

# os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"  #"If the cause is memory fragmentation maybe the environment variable 'TF_GPU_ALLOCATOR=cuda_malloc_async' will improve the situation."
#Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info. This isn't available when running in Eager mode.
# tf.compat.v1.enable_eager_execution()

# mapFeats = []
# songFeats = []
mainDir = "data/stored_feats"

# overriding train step 
# my attempt 
# it's not appropriately implemented 
# and need to fix 
class CustomTrainStep(tf.keras.Model): # via https://stackoverflow.com/questions/66472201/gradient-accumulation-with-custom-model-fit-in-tf-keras
    def __init__(self, n_gradients, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in self.trainable_variables]

    def train_step(self, data):
        self.n_acum_step.assign_add(1)

        x, y = data
        # Gradient Tape
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Calculate batch gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])
 
        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))


class AccumOptimizer(Optimizer):  # via https://stackoverflow.com/questions/55268762/how-to-accumulate-gradients-for-large-batch-sizes-in-keras
    """Inheriting Optimizer class, wrapping the original optimizer
    to achieve a new corresponding optimizer of gradient accumulation.
    # Arguments
        optimizer: an instance of keras optimizer (supporting
                    all keras optimizers currently available);
        steps_per_update: the steps of gradient accumulation
    # Returns
        a new keras optimizer.
    """
    def __init__(self, optimizer, steps_per_update=1, **kwargs):
        super(AccumOptimizer, self).__init__(name = "name", **kwargs)
        self.optimizer = optimizer
        with K.name_scope(self.__class__.__name__):
            self.steps_per_update = steps_per_update
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.cond = K.equal(self.iterations % self.steps_per_update, 0)
            self.lr = self.optimizer.lr
            self.optimizer.lr = K.switch(self.cond, self.optimizer.lr, 0.)
            for attr in ['momentum', 'rho', 'beta_1', 'beta_2']:
                if hasattr(self.optimizer, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)
                    setattr(self.optimizer, attr, K.switch(self.cond, value, 1 - 1e-7))
            for attr in self.optimizer.get_config():
                if not hasattr(self, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)
            # Cover the original get_gradients method with accumulative gradients.
            def get_gradients(loss, params):
                return [ag / self.steps_per_update for ag in self.accum_grads]
            self.optimizer.get_gradients = get_gradients
    def get_updates(self, loss, params):
        self.updates = [
            K.update_add(self.iterations, 1),
            K.update_add(self.optimizer.iterations, K.cast(self.cond, 'int64')),
        ]
        # gradient accumulation
        self.accum_grads = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        grads = self.get_gradients(loss, params)
        for g, ag in zip(grads, self.accum_grads):
            self.updates.append(K.update(ag, K.switch(self.cond, ag * 0, ag + g)))
        # inheriting updates of original optimizer
        self.updates.extend(self.optimizer.get_updates(loss, params)[1:])
        self.weights.extend(self.optimizer.weights)
        return self.updates
    def get_config(self):
        iterations = K.eval(self.iterations)
        K.set_value(self.iterations, 0)
        config = self.optimizer.get_config()
        K.set_value(self.iterations, iterations)
        return config

def getMapFeats():
    mapFeats = []
    for dir in os.listdir(mainDir):
        for item in os.listdir(os.path.join(mainDir, dir)):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".json":
                id, offsets, sr = jtf.jsonToFeats(os.path.join(mainDir, dir, item))
                mapFeats.append([id, offsets, sr])
    return mapFeats

def getSongFeats():
    songFeats = []
    for dir in os.listdir(mainDir):
        for item in os.listdir(os.path.join(mainDir, dir)):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".pkl":
                file = open(os.path.join(mainDir, dir, item), 'rb')
                data = pickle.load(file)
                songFeats.append([item[:-4], data])  # the -4 removes ".pkl"
    return songFeats

def jointlyGetMapAndSongFeats():
    # feats[]  # stored jointly for convenience: there may be multiple mapFeats per songFeats
    mapFeats = []
    songFeats = []
    index = 0
    mapCount = 0
    for dir in os.listdir(mainDir):
        mapFeats.append([])
        songFeats.append([])
        for item in os.listdir(os.path.join(mainDir, dir)):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".json":
                id, onsets, sr = jtf.jsonToFeats(os.path.join(mainDir, dir, item))
                mapFeats[index].append([id, onsets, sr])
                mapCount += 1
            if ext == ".pkl":
                file = open(os.path.join(mainDir, dir, item), 'rb')
                data = pickle.load(file)
                songFeats[index].append([item[:-4], data])  # the -4 removes ".pkl"
        index += 1

    return mapFeats, songFeats, mapCount

max_sequence_length = config.audioLengthMaxSeconds * 100  # 100 frames per second, or 10 ms per frame
def prepareFeatsForModel(mapFeats, songFeats, mapCount):
    pMapFeats = np.full((mapCount, max_sequence_length), -500)  # TODO is -500 appropriate here?
    pSongFeats = np.full((mapCount, max_sequence_length, 15, 40), -500)  # incoming as (mapCount, max_sequence_length,15,40)
    idMap = -1
    idSong = -1
    starRatings = np.empty(mapCount, dtype=float32)
    
    for i in range(len(mapFeats)):
        audioFrames = len(songFeats[i][0][1])

        # number of frames should line up with frames in audio feats
        for map in mapFeats[i]:
            # temp = 0
            idMap += 1
            onsets = map[1]  # time in ms of objects in map
            hitIndex = 0  # index of hitObject in the map (listed in variable: onsets)
            groundTruth = 0  # whether there is an object in this frame
            groundTruths = []
            for j in range(audioFrames * 10):  # for each millisecond
                if j / 10 >= max_sequence_length:
                    break
                if hitIndex < len(onsets) and int(onsets[hitIndex]) <= j:  # if there is a hit
                    hitIndex += 1
                    groundTruth = 1
                if j % 10 == 9:  # at every 10 milliseconds we summarize the 10ms frame and reset
                    groundTruths.append(groundTruth)
                    # temp += groundTruth
                    groundTruth = 0
            # print(temp, "hits")
            for j in range(len(groundTruths), max_sequence_length):
                groundTruths.append(0) # TODO this was -999, could be -1. 0 okay?
            # pMapFeats.append(tf.convert_to_tensor(groundTruths, dtype=tf.int32))
            pMapFeats[idMap] = groundTruths
            starRatings[idMap] = float32(map[2])  # star rating of map (difficulty)

        #  number of maps using the same audio file
        songRepeats = len(mapFeats[i])   # for now I am neglecting the concern about multiple maps sharing a song beind dispersed between training and test sets
        for j in range(songRepeats):
            idSong += 1
            # pSongFeats.append(tf.convert_to_tensor(songFeats[i][0][1], dtype=tf.int32))

            #TODO if issue, return to this earlier note: continue here. below trimming is incorrect and only gets one frame instead of 15 frames of context.
             
            trimmedSongFeats = songFeats[i][0][1][:min(len(songFeats[i][0][1]), max_sequence_length)]  # trim to max sequence length if applicable
            pad = []
            for k in range (max_sequence_length - len(trimmedSongFeats)):
                # print("got in with", max_sequence_length - len(trimmedSongFeats))
                pad.append(np.full((15,40), -500))
            # print(pad)
            if(len(pad) > 0):
                trimmedSongFeats = np.vstack((trimmedSongFeats, pad))
            pSongFeats[idSong] = trimmedSongFeats

    return pMapFeats, pSongFeats, starRatings

# def createModel(mapFeats, songFeats):
#     # model = keras.Sequential()
#     # # Add an Embedding layer expecting input vocab of size 1000, and
#     # # output embedding dimension of size 64.
#     # model.add(layers.Embedding(input_dim=1000, output_dim=64))

#     # # Add a LSTM layer with 128 internal units.
#     # model.add(layers.LSTM(128))

#     # # Add a Dense layer with 10 units.
#     # model.add(layers.Dense(10))

#     # model.summary()
#     model = keras.Sequential()
#     model.add(layers.Embedding(input_dim=1000, output_dim=64))

#     # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
#     model.add(layers.GRU(256, return_sequences=True))

#     # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
#     model.add(layers.SimpleRNN(128))

#     model.add(layers.Dense(10))

#     model.summary()

#     model.compile()  # TODO ensure my GPU is being used
#     print("compiled model")

def createConvLSTM():
    mapFeats, songFeats, count = jointlyGetMapAndSongFeats()
    print(count, "maps total")
    y, x, starRatings = prepareFeatsForModel(mapFeats, songFeats, count)
    #nn = models.Sequential()
    # nn.add(layers.Conv3D(?, activation = 'relu', input_dim=?)) # 
    #nn.add(layers.Conv)
    # print(np.shape(mapFeats[0][0]))

    # FULL CNN MODEL
    # convolutional layer with 10 filter kernels that are 7-wide in time and 3-wide in frequency. ReLU.
    # 1D max-pooling only in the frequency dimension, with a width and stride of 3
    # convolutional layer with 20 filter kernels that are 3-wide in time and 3-wide in frequency. ReLU.
    # 1D max-pooling only in the frequency dimension, with a width and stride of 3
    # fully connected layerr with ReLU activation functions and 256 nodes
    # fully connected layerr with ReLU activation functions and 128 nodes

    # C-LSTM model
    # convolutional layer with 10 filter kernels that are 7-wide in time and 3-wide in frequency. ReLU.
    # 1D max-pooling only in the frequency dimension, with a width and stride of 3
    # convolutional layer with 20 filter kernels that are 3-wide in time and 3-wide in frequency. ReLU.
    #? 1D max-pooling only in the frequency dimension, with a width and stride of 3
    # Output of second conv. layer is a 2D tensor for me. Flatten it along the (channel and) frequency axis. (preserving temporal dimension)
    # LSTM with 200 nodes
    # LSTM with 200 nodes
    # fully connected ReLU layer with dimension 256
    # fully connected ReLU layer with dimension 128
    # this model is trained using 100 unrollings for backpropagation through time.

    learning_rate = 0.01
    conv1d_strides = 12
    conv1d_1_strides = 12   
    conv1d_filters = 4
    hidden_units = 200 # 42
    # padding_value = -999
    # seq_length_cap = 30000  # 30000 frames = 300 seconds = 5 minutes

    # xpad = np.full((17, seq_length_cap, 3), fill_value=padding_value)
    # for s, x in enumerate(songFeats):
    #     seq_len = x.shape[0]
    #     xpad[s, 0:seq_len, :] = x
    # print(xpad)

    print(x.shape, y.shape, starRatings.shape)  # currently (17, 24000, 15, 40) (17, 24000) (17,)
    

    x = reshape(x, (len(x), len(x[0]), len(x[0][0]), len(x[0][0][0]),1))  # 17, 24000, 15, 40, 1
    y = y.reshape((len(y),len(y[0]),1,1))
    x = x.astype(float32)
    starRatings = starRatings.reshape((len(starRatings), 1, 1))

    # x = reshape(x, (len(x), 100, len(x[0][0]), 1,1))  # 17, 100, 40, 1
    # y = y.reshape((len(y), 100 ,1,1))   
    # x = x.astype(float32)
    
    x = x.reshape((len(x), len(x[0]), 1, len(x[0][0]), len(x[0][0][0])))  # 17, 24000, 1, 15, 40
    print(x.shape, y.shape)


    # Create Sequential Model ###########################################
    clear_session()
    # model = Sequential()
    # model.add(TimeDistributed(Conv2D(10, (7,3),activation='relu', padding='same',input_shape=(25,40,1,1),data_format='channels_first'))) # shape is 12+1+12 frames x 40 frequency bands x 1 row x 1 col
    # model.add(TimeDistributed(MaxPool2D(pool_size=(1,3), padding='same')))
    # model.add(TimeDistributed(Conv2D(20, (3,3),activation='relu', padding='same')))
    # model.add(TimeDistributed(MaxPool2D(pool_size=(1,3), padding='same')))
    # model.add(TimeDistributed(Flatten())) # see above notes, does this overly flatten temporal?
    # model.add(LSTM(hidden_units, return_sequences=True))
    # model.add(Dropout(0.5, noise_shape=(None,1,hidden_units)))  # is this shape correct?
    # model.add(LSTM(hidden_units, return_sequences=True))  # TODO do we want return_sequences again, really?
    # model.add(Dropout(0.5, noise_shape=(None,1,hidden_units))) 
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5)) 
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5)) 

    input = tf.keras.Input(shape=(max_sequence_length,1,15,40))

    # def context(x):
    #     padding = np.full((1,1,40), -500, dtype=float32)
    #     context = 7
    #     window = 2*context + 1 # prepend and append context
    #     # want to create input_shape=(max_sequence_length,15,1,40) from (max_sequence_length, 1, 1, 40)
    #     out = np.zeros((max_sequence_length,window,1,40))
    #     for i in range (max_sequence_length):
    #         bookended = np.zeros((window,1,40), dtype=float32)
    #         for j in range (context*-1, context+1):
    #             indexToGet = i + j  # if at start of audio this is negative in first half, if at end this is out of bounds positive in second half
    #             if indexToGet < 0 or indexToGet >= max_sequence_length:
    #                 bookended[j + context] = padding
    #             else:
    #                 sess = tf.compat.v1.Session()
    #                 with sess.as_default():
    #                     temp = x[i + j]
    #                     eval = temp.eval()
    #                     bookended[j + context] = eval  # if all else fails then just do this in audio processing. will use 14x more storage.
                        
    #                     # bookended[j + context] = x[i + j].eval()
    #         out[i] = bookended
    #     tensor_out = tf.constant(out)
    #     return out

            

        

    

    # base_maps = tf.keras.layers.Lambda(context)(input)
    base_maps = TimeDistributed(Conv2D(10, (7,3),activation='relu', padding='same',data_format='channels_last'))(input) # TODO is 25 being scanned corectly?
    base_maps = TimeDistributed(MaxPool2D(pool_size=(1,3), padding='same'))(base_maps)
    base_maps = TimeDistributed(Conv2D(20, (3,3),activation='relu', padding='same',data_format='channels_last'))(base_maps)
    base_maps = TimeDistributed(MaxPool2D(pool_size=(1,3), padding='same'))(base_maps)
    base_maps = TimeDistributed(Flatten())(base_maps) # see above notes, does this overly flatten temporal?

    # sequence = tf.keras.Input(shape=(max_sequence_length, hidden_units))  # TODO core issue? is this shape sane? Is LSTM getting entire length of audio at once?
    # part1 = base_maps(sequence)

    # base_maps = tf.keras.Input(shape=(max_sequence_length, hidden_units))(base_maps)

 #   starRatingFeat = tf.keras.Input(shape=(1, 1))

    # merged = concatenate([base_maps, starRatingFeat])

 #   merged = tf.keras.layers.Concatenate()([base_maps, starRatingFeat])

    base_maps = LSTM(hidden_units, return_sequences=True, input_shape=(25,200))(base_maps)#(merged)  #TODO input shape? Needed? Correct? Used?
    base_maps = Dropout(0.5, noise_shape=(None,1,hidden_units))(base_maps)  # is this shape correct? TODO fix
    base_maps = LSTM(hidden_units, return_sequences=True)(base_maps)  # TODO do we want return_sequences again, really?
    base_maps = Dropout(0.5, noise_shape=(None,1,hidden_units))(base_maps)   # TODO noise shape may be incorrect now after shape changes
    base_maps = Dense(256, activation='relu')(base_maps)
    base_maps = Dropout(0.5)(base_maps)
    base_maps = Dense(128, activation='relu')(base_maps)
    base_maps = Dropout(0.5)(base_maps) 

    base_maps = Dense(1, activation='sigmoid')(base_maps)

    epochs = 1
    gradients_per_update = 4  # i.e., number of batches to accumulate gradients before updating. Effective batch size after gradient accumulation is this * batch size.
    batch_size = 10

    ga_model = CustomTrainStep(n_gradients=gradients_per_update, inputs=[input], outputs=[base_maps])
#   ga_model = CustomTrainStep(n_gradients=gradients_per_update, inputs=[input, starRatingFeat], outputs=[base_maps])

    # bind all
    ga_model.compile(  #ga for gradient accumulation
        loss = 'binary_crossentropy',
        # metrics = ['accuracy'],
        optimizer = tf.keras.optimizers.SGD(momentum=0.01, nesterov=True, learning_rate=learning_rate),
        metrics = tf.keras.metrics.AUC(curve='PR') )

    history = ga_model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, )
#   history = ga_model.fit([x, starRatings], y, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, )
    print(ga_model.summary())

    
    
    
    # opt = keras.optimizers.SGD(momentum=0.01, nesterov=True, learning_rate=learning_rate)
    # # opt = AccumOptimizer(opt, 9) # number is accumulative steps, TODO

    # opt = runai.ga.keras.optimizers.Optimizer(opt, steps=9)

    # # model.compile(loss='mse', optimizer=opt)
    # # model.fit(x_train, y_train, epochs=10, batch_size=10)


    # model.compile(optimizer=opt, loss='binary_crossentropy')#, loss='binary_crossentropy'))# loss=error_to_signal, metrics=[error_to_signal]) previously used Adam
    

    # test_size = 0.2
    # epochs = 4
    # batch_size = 10  # 43 works with the defragmentation option enabled. This is a very small improvement.
    # #35 works, 42 is OOM. No matter what number, dedicated GPU memory usage goes near max.
    # #25 # ? decreasing this from total map count makes more parts per epoch. (frequently weights are updated)
    # # batch size of 1 runs okay. Recently larger numbers crash (exceeding mem?). 5 throws CUDNN_STATUS_NOT_SUPPORTED, 50 throws plain mem error (OOM)

    # print(x.shape, y.shape)
    # print("begin fit")
    # history = model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=test_size)
    # # model.build((17,1,30000,40,1))
    # print(model.summary())
    # print(history)

    ga_model.save("models/onset")
    # TODO save training history to be viewed later

    with open('models/history.pickle', 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # indices = tf.range(start=0, limit=tf.shape(songFeats)[0], dtype=tf.int32)
    # shuffled_indices = tf.random.shuffle(indices)

    # X_random = tf.gather(songFeats, shuffled_indices)
    # Y_random = tf.gather(mapFeats, shuffled_indices)
    # songFeats = tf.convert_to_tensor(songFeats, dtype=tf.int32)
    # x = songFeats.reshape(len(songFeats),1)

    # mapFeats = np.asarray(mapFeats).astype('int32')
    # y = mapFeats.reshape(len(mapFeats), 1)

    
    # shuffled_indices = np.random.permutation(len(songFeats)) 
    # X_random = tf.gather(songFeats, shuffled_indices)
    # y_random = tf.gather(mapFeats, shuffled_indices)

    # mapFeats = tf.ragged.constant(mapFeats)
    # print("begin fit")
    # model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=test_size, verbose=1)
    
    
    # model.fit(np.asarray(songFeats).astype('float32'), np.asarray(mapFeats).astype('float32'), epochs=epochs, batch_size=batch_size, validation_split=test_size)
    #model.save('test_model.h5')

    print("got to end")

# def basicModel():
    # mapFeats = getMapFeats()
    # songFeats = getSongFeats()  # array of [mels pkl title, mels pkl data] pairs, one per song folder
    # mapFeats, songFeats, totalMaps = jointlyGetMapAndSongFeats()
    # mapFeats, songFeats = prepareFeatsForModel(mapFeats, songFeats, totalMaps)

    # print(mapFeats.shape)
    # print(songFeats.shape)
    # mapFeats = sorted(mapFeats, key=lambda x: x[0])  # sort by id (makeshift title: song-mapper-diff)
    # songFeats = sorted(songFeats, key=lambda x: x[0])  # similar to above
    # no need to do above sorting with new implementation, they already line up.
    # sloppy but at this point we expect the IDs to basically line up. But notably there are usually more mapFeats than songFeats: Some maps share a song.
    # print(mapFeats[8][0][0])
    # print(songFeats[8][0][0])

    # for i in range(14):
    #     print(len(mapFeats[i]), len(songFeats[i]))
    # print(songFeats[0][0][1].shape)
    # createConvLSTM()

    # createModel(mapFeats, songFeats)
    # print(songFeats[0][1][82])
    # print(mapFeats.shape)
    # print()
    # print(songFeats[0][1].shape)

# basicModel()




createConvLSTM()

# model = tf.keras.models.load_model("models/onset")
# audioFile = "sample_maps/1061593 katagiri - Urushi/audio.wav"
name = "Urushi"
# prediction = controllers.onset_predict.makePredictionFromAudio(model, audioFile) #TODO
# processedPrediction = controllers.onset_predict.processPrediction(prediction) #TODO peak picking, etc. Must create a list of essentially booleans: object or no object at each frame.

#result = onset_generate_taiko_map.convertOnsetPredictionToMap(prediction, audioFile, name) #TODO
#print(result)

print("got to end of file")