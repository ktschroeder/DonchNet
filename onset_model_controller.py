import os
import feature_extraction.map_json_to_feats as jtf
import pickle


mapFeats = []
songFeats = []

def getMapFeats():
    mainDir = "./data/json"
    # mapFeats = []
    for item in os.listdir(mainDir):
        id, offsets = jtf.jsonToFeats(os.path.join(mainDir, item))
        mapFeats.append([id, offsets])
    return mapFeats

def getSongFeats():
    mainDir = "./data/mels"
    # songFeats = []
    for item in os.listdir(mainDir):
        file = open(os.path.join(mainDir, item), 'rb')
        data = pickle.load(file)
        songFeats.append([item[:-4], data])  # the -4 removes ".pkl"
    return songFeats


def basicModel():
    mapFeats = getMapFeats()
    songFeats = getSongFeats()  # array of [mels pkl title, mels pkl data] pairs, one per song folder
    mapFeats = sorted(mapFeats, key=lambda x: x[0])  # sort by id (makeshift title: song-mapper-diff)
    songFeats = sorted(songFeats, key=lambda x: x[0])  # similar to above
    # sloppy but at this point we expect the IDs to basically line up. But notably there are usually more mapFeats than songFeats: Some maps share a song.
    print(mapFeats[0][0])
    print(songFeats[0][0])
    

    #print(songFeats[0][1][82])

basicModel()