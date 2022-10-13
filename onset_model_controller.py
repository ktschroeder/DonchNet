import os
import feature_extraction.map_json_to_feats as jtf

def getMapFeats():
    mainDir = "./data/json"
    mapFeats = []
    for item in os.listdir(mainDir):
        mapFeats.append(jtf.jsonToFeats(item))

def getSongFeats():
    



def basicModel():
    mapFeats = getMapFeats()
    songFeats = getSongFeats()
