import json, unicodedata, re

def makeSafeFilename(name):  # taken from Django's slugify function
    # Normalizes string, converts to lowercase, removes non-alpha characters, and converts spaces to underscores.
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = re.sub(r'[^\w\s-]', '', name.lower())
    return re.sub(r'[-\s]+', '_', name).strip('-_')

def jsonToFeats(file):
    data = open(file,'rt').read()
    j = json.loads(data)
    onsets = []
    for i in range (len(j["hitobjects"])):
        onsets.append(j["hitobjects"][i]["time"])  # offset in ms, currently ignoring type of object
    # songName = j["metadata"]["Title"].strip()
    # mapper = j["metadata"]["Creator"].strip()
    # diffName = j["metadata"]["Version"].strip()
    sr = j["sr"]
    
    id = makeSafeFilename(j["metadata"]["Title"].strip()) + '-' + makeSafeFilename(j["metadata"]["Creator"].strip()) + '-' + makeSafeFilename(j["metadata"]["Version"].strip())
    return id, onsets, sr


def jsonToFeatsColor(fileOrJson):  # can be file name of json or actual json
    j = None
    try:
        j = json.loads(fileOrJson)  # this will throw an exception if it is a file
    except ValueError as e:
        data = open(fileOrJson,'rt').read()
        # print(fileOrJson)
        j = json.loads(data)
    assert(j)

    onsets = []
    notes = []
    for i in range (len(j["hitobjects"])):
        onsets.append(round(float((j["hitobjects"][i]["time"]))))  # offset in ms, currently ignoring type of object
        notes.append(j["hitobjects"][i]["hitsound"])
    sr = j["sr"]
    
    id = makeSafeFilename(j["metadata"]["Title"].strip()) + '-' + makeSafeFilename(j["metadata"]["Creator"].strip()) + '-' + makeSafeFilename(j["metadata"]["Version"].strip())
    return id, onsets, notes, sr