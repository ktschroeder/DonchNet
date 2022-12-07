from pydub import AudioSegment, effects

def processAndExport(sound, targetPath):
    sound = sound.set_channels(1)  # convert to mono
    sound = effects.normalize(sound)  # normalize volume
    sound.export(targetPath, format="wav", bitrate="192k")  # export at 192 kbps
    return

def mp3ToWav(mp3File, targetPath):
    # print("mp3")
    sound = AudioSegment.from_mp3(mp3File)
    processAndExport(sound, targetPath)
    return 1

def oggToWav(oggFile, targetPath):
    # print("ogg")
    sound = AudioSegment.from_ogg(oggFile)
    processAndExport(sound, targetPath)
    return 1