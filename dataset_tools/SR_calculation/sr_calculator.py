# given a beatmap json, should calculate an SR identically to the osu system's calculation for taiko SR.
# 
# 3 skills: Stamina, colour, rhythm. Peak strain for each is measured in each 400ms segment of a map.
#
# Difficulty calculation algorithm: Copyright (c) ppy Pty Ltd <contact@ppy.sh>. Licensed under the MIT Licence.
#
#
import rhythm, colour, stamina

final_multiplier = 0.0625
rhythm_skill_multiplier = 0.2 * final_multiplier
colour_skill_multiplier = 0.375 * final_multiplier
stamina_skill_multiplier = 0.375 * final_multiplier

# public double ColourDifficultyValue => colour.DifficultyValue() * colour_skill_multiplier;
# public double RhythmDifficultyValue => rhythm.DifficultyValue() * rhythm_skill_multiplier;
# public double StaminaDifficultyValue => stamina.DifficultyValue() * stamina_skill_multiplier;

def norm(p, values):  # TODO look at papers mentioning osu's SR algorithms
    x = 0
    for i in values:
        x += pow(i, p)
    return pow(x, 1 / p)


for obj in hitObjects:
    rhythm.process(obj)
    colour.process(obj)
    stamina.process(obj)

    peaks = []

    colourPeaks = colour.peaks
    rhythmPeaks = rhythm.peaks
    staminaPeaks = stamina.peaks

    for i in range (len(colourPeaks)):
        colourPeak = colourPeaks[i] * colour_skill_multiplier
        rhythmPeak = rhythmPeaks[i] * rhythm_skill_multiplier
        staminaPeak = staminaPeaks[i] * stamina_skill_multiplier

        peak = norm(1.5, colourPeak, staminaPeak)  # condense 3 skill peaks into 1 summary peak
        peak = norm(2, peak, rhythmPeak)

        if (peak > 0):
            peaks.Add(peak)
    
    difficulty = 0
    weight = 1
    peaks.sort(reverse=1)

    for strain in peaks:
        difficulty += strain * weight
        weight *= 0.9

    print("SR " + difficulty)
