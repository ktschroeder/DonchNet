import glob, os

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
    try:
        #os.remove(file)
        return 1
    except:
        print(f"Failed to delete file: {file}")
        return 0

processed = 0
mscFilesDeleted = 0
osuFilesDeleted = 0
DEBUG = 5300  # TODO remove debug
print("Provide address to collective Songs folder to simplify. Everything will be deleted except taiko .osu files and necessary audio files (and directories).")
dirInput = input("")
couldNotAutoProcess = []
for root, dirs, files in os.walk(dirInput):
    for dir in dirs:
        if processed < DEBUG:  # TODO remove debug
            processed += 1
            continue

        os.chdir(os.path.join(root, dir))

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
            print(f"Could not process {dir}: zero or multiple taiko map audio files")
            couldNotAutoProcess.append(dir)
            processed += 1
            continue
            

        #assert audioFilename, f"no audio filename found among all osu files: {dir}"  # disabling this is more convenient. See e.g. map #16458

        for file in glob.glob("*"):
            if file != audioFilename and file not in glob.glob("*.osu"):
                mscFilesDeleted += delete(file)
                
        for file in glob.glob("*.osu"):
            if not isTaiko(file):
                osuFilesDeleted += delete(file)

        processed += 1
        if(not processed % 100):
            print(f"processed {processed} song folders...")

print(f"Deleted {osuFilesDeleted} non-taiko .osu files and {mscFilesDeleted} miscellaneous files.")
if len(couldNotAutoProcess) > 0:
    print(f"{len(couldNotAutoProcess)} song folders could not be automatically processed and must be manually processed. They are:")
    # for item in couldNotAutoProcess:
    #     print(item)

