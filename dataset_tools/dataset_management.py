import os
import map_to_json_converter

def makeJsons():
    made = 0
    mainDir = "C:/Users/Admin/Documents/small_taiko_dataset"  # TODO temporary
    for songFolder in os.listdir(mainDir):
        for item in os.listdir(os.path.join(mainDir, songFolder)):
            ext = os.path.splitext(item)[-1].lower()
            if ext == ".osu":
                made += map_to_json_converter.mapToJson(os.path.join(mainDir, songFolder, item))
    print(f"made {made} JSONs")

makeJsons()