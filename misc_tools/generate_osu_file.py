import random, os

def generateOsuFile(onsets, name, starRating, audioFile):
    # TODO automatic snapping? There is no well-defined starting offset, also red barlines can change and appear several times normally. Maybe leave this to be done manually

    randomID = random.randrange(999999)
    out = f'''osu file format v14

[General]
AudioFilename: {os.path.basename(audioFile)}
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
Creator:{random.randrange(999999)}
Version:Target SR {starRating} (onsets) {random.randrange(999999)}
Source:{random.randrange(999999)}
Tags:DonNet, {random.randrange(999999)}
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
0,0,"bg.jpg",0,0
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

    hits = []
    for i in onsets:
        hits.append(f"288,160,{i},1,0,0:0:0:0:\n")
    out = out + ''.join(hits)
    print(f"{len(onsets)} onsets created for {name}")

    return out