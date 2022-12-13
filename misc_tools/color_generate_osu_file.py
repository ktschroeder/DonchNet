import random
from dataset_tools import map_to_json_converter
from feature_extraction import map_json_to_feats

def getOnsets(osuFile):
    json = map_to_json_converter.mapToJson(osuFile)
    _, onsets, _, _ = map_json_to_feats.jsonToFeatsColor(json)
    return onsets


def generateOsuFile(prediction, mapfile, name):

    # determine onsets from original (onset) map
    onsets = getOnsets(mapfile)
    assert(len(onsets) == len(prediction))

    out = f'''osu file format v14

[General]
AudioFilename: audio.mp3
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Normal
StackLeniency: 0.7
Mode: 1
LetterboxInBreaks: 0
WidescreenStoryboard: 1

[Editor]
DistanceSpacing: 2.8
BeatDivisor: 2
GridSize: 32
TimelineZoom: 2

[Metadata]
Title:{name}
TitleUnicode:{name}
Artist:DonNet
ArtistUnicode:DonNet
Creator:DonNet
Version:DonNet
Source:DonNet
Tags:DonNet
BeatmapID:0
BeatmapSetID:-1

[Difficulty]
HPDrainRate:6
CircleSize:5
OverallDifficulty:5
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1

[Events]
//Background and Video events
//Break Periods
//Storyboard Layer 0 (Background)
//Storyboard Layer 1 (Fail)
//Storyboard Layer 2 (Pass)
//Storyboard Layer 3 (Foreground)
//Storyboard Layer 4 (Overlay)
//Storyboard Sound Samples

[TimingPoints]
10,400,4,1,0,70,1,0

[HitObjects]
'''

    # expecting prediction in form: [[0,0,1,0],[0,1,0,0], ... ] where each quadruplet is the color of the respective onset
    assert(len(prediction[0]) == 4)
    hits = []
    for i in range(len(onsets)):
        hitSound = -1
        if prediction[i][0] == 1:    # don
            hitSound = 0
        elif prediction[i][1] == 1:  # kat
            hitSound = 8
        elif prediction[i][2] == 1:  # fdon
            hitSound = 4
        elif prediction[i][3] == 1:  # fkat
            hitSound = 12
        else:
            assert(None)

        hits.append(f"288,160,{onsets[i]},1,{hitSound},0:0:0:0:\n")
    out = out + ''.join(hits)
    print(f"{len(prediction)} colors created for {name}")

    return out