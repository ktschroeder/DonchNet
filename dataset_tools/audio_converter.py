from pydub import AudioSegment

def mp3ToWav(mp3File, targetPath):
    # print("mp3")
    sound = AudioSegment.from_mp3(mp3File)
    sound.export(targetPath, format="wav")
    return 1

def oggToWav(oggFile, targetPath):
    # print("ogg")
    sound = AudioSegment.from_ogg(oggFile)
    sound.export(targetPath, format="wav")
    return 1