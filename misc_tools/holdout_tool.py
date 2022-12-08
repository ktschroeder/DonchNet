import os
import random
import shutil

source = 'data/stored_feats'
dest = 'data/holdout_feats'
folders = os.listdir(source)
nfiles = len(folders) // 10
print(nfiles)

# for folder in random.sample(folders, nfiles):
#     shutil.move(os.path.join(source, folder), dest)