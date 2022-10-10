import glob, os

def getAudioFilename(file):
    f = open(file, 'rt')
    for line in f:
        try:
            if line.index("AudioFilename: ") == 0:
                return line.split(' ')[1].strip()
        except:
            continue
    raise Exception(".osu file with no audio filename")

def isTaiko(file):
    f = open(file, 'rt')
    for line in f:
        try:
            if line.index("Mode: ") == 0:
                isTaiko = line.split(' ')[1].strip() == '1'  # mode 1 corresponds to taiko mode
                return isTaiko
        except:
            continue
    raise Exception(".osu file with no mode")

def delete(file):
    try:
        os.remove(file)
        return 1
    except:
        print(f"Failed to delete file: {file}")
        return 0


mscFilesDeleted = 0
osuFilesDeleted = 0
print("Provide address to collective Songs folder to simplify. Everything will be deleted except taiko .osu files and necessary audio files (and directories).")
dirInput = input("")
for root, dirs, files in os.walk(dirInput):
    for dir in dirs:
        os.chdir(os.path.join(root, dir))

        audioFilename = ""
        for file in glob.glob("*.osu"):
            newAudioFilename = getAudioFilename(file)
            if newAudioFilename != audioFilename:
                assert not audioFilename, f"two different audio filenames in one map folder: {dir}"
            audioFilename = newAudioFilename

        assert audioFilename, f"no audio filename found among all osu files: {dir}"

        for file in glob.glob("*"):
            if file != audioFilename and file not in glob.glob("*.osu"):
                mscFilesDeleted += delete(file)
                
        for file in glob.glob("*.osu"):
            if not isTaiko(file):
                osuFilesDeleted += delete(file)

print(f"Deleted {osuFilesDeleted} non-taiko .osu files and {mscFilesDeleted} miscellaneous files.")
