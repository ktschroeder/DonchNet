import os
import feature_extraction.map_json_to_feats as jtf
import pickle


# mapFeats = []
# songFeats = []
mainDir = "data/stored_feats"

def getMapFeats():
    mapFeats = []
    for dir in os.listdir(mainDir):
        for item in os.listdir(os.path.join(mainDir, dir)):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".json":
                id, offsets = jtf.jsonToFeats(os.path.join(mainDir, dir, item))
                mapFeats.append([id, offsets])
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
    for dir in os.listdir(mainDir):
        mapFeats.append([])
        songFeats.append([])
        for item in os.listdir(os.path.join(mainDir, dir)):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".json":
                id, offsets = jtf.jsonToFeats(os.path.join(mainDir, dir, item))
                mapFeats[index].append([id, offsets])
            if ext == ".pkl":
                file = open(os.path.join(mainDir, dir, item), 'rb')
                data = pickle.load(file)
                songFeats[index].append([item[:-4], data])  # the -4 removes ".pkl"
        index += 1
    return mapFeats, songFeats

def createModel(mapFeats, songFeats):



def basicModel():
    # mapFeats = getMapFeats()
    # songFeats = getSongFeats()  # array of [mels pkl title, mels pkl data] pairs, one per song folder
    mapFeats, songFeats = jointlyGetMapAndSongFeats()
    # mapFeats = sorted(mapFeats, key=lambda x: x[0])  # sort by id (makeshift title: song-mapper-diff)
    # songFeats = sorted(songFeats, key=lambda x: x[0])  # similar to above
    # no need to do above sorting with new implementation, they already line up.
    # sloppy but at this point we expect the IDs to basically line up. But notably there are usually more mapFeats than songFeats: Some maps share a song.
    # print(mapFeats[8][0][0])
    # print(songFeats[8][0][0])

    createModel(mapFeats, songFeats)
    #print(songFeats[0][1][82])

basicModel()