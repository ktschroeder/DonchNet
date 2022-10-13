# Difficulty calculation algorithm: Copyright (c) ppy Pty Ltd <contact@ppy.sh>. Licensed under the MIT Licence.

class TaikoDifficultyHitObject:
    monoDifficultyHitObjects = []  # list of notes of same color as this
    MonoIndex = -1  # index of this in monoDifficultyHitObjects
    noteDifficultyHitObjects = []  # list of all regular or finisher notes (no sliders/spinners)
    NoteIndex = -1  # index of this in noteDifficultyHitObjects
    Rhythm = []  # rhythm info for this (new object?)
    Colour = []  # colour info for this (new object?)
