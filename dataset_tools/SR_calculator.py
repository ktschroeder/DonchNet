from peace_performance_python.objects import Beatmap, Calculator

def calculate(beatmap: Beatmap, calculator: Calculator):
    return calculator.calculate(beatmap)

# this calculates previous SR algo values ("local SR" on my osu client)
def calculateSR(path):
    beatmap = Beatmap(path)
    c = Calculator(mode=1).calculate(beatmap)  # mode 1 corresponds to taiko
    return c.stars