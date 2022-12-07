import os
import shutil
mainDir = "C:/Users/Admin/Documents/GitHub/taiko_project/data/prediction_maps"

for dir in os.listdir(mainDir):
    zip = shutil.make_archive(os.path.join(mainDir, dir), 'zip', os.path.join(mainDir, dir))
    os.rename(os.path.join(mainDir, zip), os.path.join(mainDir, zip)[:-3] + "osz")
