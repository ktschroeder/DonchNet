# globals etc.
import numpy as np

audioLengthMaxSeconds = 240
colorUnrollings = 64
colorOnsetMax = 1000  # onsets after this index in each map will be excluded
colorAudioBookendLength = 3  # length of prepend and append bookends for audio analysis in color model
featureMainDirectory = "data/stored_feats"
datasetMainDirectory = "C:/Users/Admin/osu!/Songs disabled"
# datasetMainDirectory = "C:/Users/Admin/Documents/adhoc_thing"  # WARNING this will screw up normalization and bandMeans/bandStdevs. Only for debugging

def determinePad():
    stdevs = np.loadtxt("data/misc/bandStdevs.txt")
    base = -500  # this is the value associated with no sound in a frequency band, as determined by the audio conversion functions
    minStdev = np.amin(stdevs)  # the lowest standard deviation will result in the lowest pad
    pad = base / minStdev  # this should be about equal to the lowest value in the full dataset across all songs and frequency bands
    return pad

pad = determinePad()
print(f"Determined pad to be {pad}.")


# via .osu standard, hitsounds: 
# don: 0
# kat: 2/8/10
# fdon: 4
# fkat: 6/12/14