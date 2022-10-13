# Main songs folder before razor: 47.0 GB. After razor: 26.4 GB (18862 beatmaps, 6472 song folders)
# special cases from initial dataset: 
# 1236247 Iambic 9 Poetry: has no taiko maps
# 71856 murmur twins: has two different song audiofiles for different taiko maps. I split these into a second song folder.

import glob, os, shutil

def getAudioFilename(file, enc="utf8"):
    try:
        f = open(file, 'rt', encoding=enc)
        for line in f:
            try:
                if line.index("AudioFilename:") == 0:
                    return line.split(':')[1].strip()
            except:
                continue
        raise Exception(f".osu file with no audio filename: {file}")
    except UnicodeDecodeError:
        if enc == "utf8":
            return getAudioFilename(file, "utf16")
        if enc == "utf16":
            print(f"getAudioFileName: utf8 and utf16 decoding both failed in file: {file}")
    

def isTaiko(file, enc="utf8"):
    try:
        f = open(file, 'rt', encoding=enc)
        for line in f:
            try:
                if line.index("Mode:") == 0:
                    isTaiko = line.split(':')[1].strip() == '1'  # mode 1 corresponds to taiko mode
                    return isTaiko
            except:
                continue
        raise Exception(f".osu file with no mode: {file}")
    except UnicodeDecodeError:
        if enc == "utf8":
            return getAudioFilename(file, "utf16")
        if enc == "utf16":
            print(f"getAudioFileName: utf8 and utf16 decoding both failed in file: {file}")


def delete(file):
    if os.path.isdir(file):
        return 0
    try:
        os.remove(file)
        return 1
    except:
        print(f"Failed to delete file: {file}")
        return 0

def deleteDir(dir):
    try:
        shutil.rmtree(dir)
        return 1
    except:
        print(f"Failed to delete directory: {dir}")
        return 0

processed = 0
mscFilesDeleted = 0
osuFilesDeleted = 0
directoriesDeleted = 0
DEBUG = 0  # TODO remove debug
print("Provide address to collective Songs folder to simplify. Everything will be deleted except taiko .osu files and necessary audio files.")
dirInput = input("")
couldNotAutoProcess = []
#for root, dirs, files in os.walk(dirInput):
for songFolder in os.listdir(dirInput):
    

    if processed < DEBUG:  # TODO remove debug
        processed += 1
        continue

    os.chdir(os.path.join(dirInput, songFolder))

    audioFilenamesFound = 0
    audioFilename = ""
    for file in glob.glob("*.osu"):
        if not isTaiko(file):
            continue
        newAudioFilename = getAudioFilename(file)
        if newAudioFilename != audioFilename:
            audioFilenamesFound += 1
            audioFilename = newAudioFilename
            #assert not audioFilename, f"two different audio filenames in one map folder among taiko maps: {dir}"
    if audioFilenamesFound != 1:
        print(f"Could not process {songFolder}: zero or multiple taiko map audio files")
        couldNotAutoProcess.append(songFolder)
        processed += 1
        continue
        

    #assert audioFilename, f"no audio filename found among all osu files: {dir}"  # disabling this is more convenient. See e.g. map #16458

    for file in glob.glob("*"):
        if file != audioFilename and file not in glob.glob("*.osu"):
            mscFilesDeleted += delete(file)
            
    for file in glob.glob("*.osu"):
        if not isTaiko(file):
            osuFilesDeleted += delete(file)

    for item in os.listdir():  # there shouldn't be anything useful in secondary directories, this is presumably storyboards and other miscellanea
        if os.path.isdir(item):
            directoriesDeleted += deleteDir(item)

    processed += 1
    if(not processed % 250):
        print(f"processed {processed} song folders...")

print(f"Deleted {osuFilesDeleted} non-taiko .osu files, {mscFilesDeleted} miscellaneous files, and {directoriesDeleted} miscellaneous folders (and their contents).")
if len(couldNotAutoProcess) > 0:
    print(f"{len(couldNotAutoProcess)} song folders could not be automatically processed and must be manually processed. They are:")
    for item in couldNotAutoProcess:
        print(item)

