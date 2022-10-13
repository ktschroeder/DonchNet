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
    id = makeSafeFilename(j["metadata"]["Title"].strip()) + '-' + makeSafeFilename(j["metadata"]["Creator"].strip()) + '-' + makeSafeFilename(j["metadata"]["Version"].strip())
    return id, onsets