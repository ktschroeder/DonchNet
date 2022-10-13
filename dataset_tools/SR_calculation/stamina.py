# Difficulty calculation algorithm: Copyright (c) ppy Pty Ltd <contact@ppy.sh>. Licensed under the MIT Licence.
import TaikoDifficultyHitObject

SkillMultiplier = 1.1
StrainDecayBase = 0.4
currentStrain = 0

def process(hitObject):
    print()

def strainValueAt(hitObject):
    currentStrain *= strainDecay(hitObject.DeltaTime)
    currentStrain += strainValueOf(hitObject) * SkillMultiplier
    return currentStrain

def strainValueOf(hitObject):
    return staminaEvaluateDiffOf(hitObject)

def staminaEvalDiffOf(current):
    if (current.BaseObject is not Hit):
        return 0.0

    # Find the previous hit object hit by the current key, which is two notes of the same colour prior.
    keyPrevious = current.PreviousMono(1)

    if (not keyPrevious):
        # There is no previous hit object hit by the current key
        return 0.0

    objectStrain = 0.5; # Add a base strain to all objects
    objectStrain += speedBonus(current.StartTime - keyPrevious.StartTime)
    return objectStrain