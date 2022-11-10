import json, os, unicodedata, re
import SR_calculator

def makeSafeFilename(name):  # taken from Django's slugify function
    # Normalizes string, converts to lowercase, removes non-alpha characters, and converts spaces to underscores.
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = re.sub(r'[^\w\s-]', '', name.lower())
    return re.sub(r'[-\s]+', '_', name).strip('-_')


def mapToJson(file, targetDir, enc="utf8"):  # "C:/Users/Admin/Documents/GitHub/taiko-project/dataset_tools/DJ SIMON - 321STARS (Zelos) [Taiko Oni].osu"

    osu = ""
    
    try:
        osu = open(file,'rt', encoding=enc).readlines()
    except UnicodeDecodeError:
        if enc == "utf8":
            return mapToJson(file, targetDir, "utf16")
        if enc == "utf16":
            print(f"mapToJson: utf8 and utf16 decoding both failed in file: {file}")

    out = {}

    def get_line(phrase):
        for num, line in enumerate(osu, 0):
            if phrase in line:
                return num

    out['general'] = {}
    out['metadata'] = {}
    out['difficulty'] = {}
    out['timingpoints'] = []
    out['hitobjects'] = []

    general_line = get_line('[General]')
    editor_line = get_line('[Editor]')
    metadata_line = get_line('[Metadata]')
    difficulty_line = get_line('[Difficulty]')
    events_line = get_line('[Events]')
    timing_line = get_line('[TimingPoints]')
    hit_line = get_line('[HitObjects]')

    general_list = osu[general_line:editor_line-1]
    metadata_list = osu[metadata_line:difficulty_line-1]
    difficulty_list = osu[difficulty_line:events_line-1]
    timingpoints_list = osu[timing_line:hit_line-1]
    hitobject_list = osu[hit_line:]

    for item in general_list:
        if ':' in item:
            item = item.split(':')
            out['general'][item[0]] = item[1]

    for item in metadata_list:
        if ':' in item:
            item = item.split(':')
            out['metadata'][item[0]] = item[1]

    for item in difficulty_list:
        if ':' in item:
            item = item.split(':')
            out['difficulty'][item[0]] = item[1]

    for item in timingpoints_list:
        if ',' in item:
            item = item.split(',')
            point = {
            'offset':item[0],
            'beatlength':item[1],
            'meter':item[2]
            }
            out['timingpoints'].append(point)

    for item in hitobject_list:
        if ',' in item:
            item = item.split(',')
            isSlider = item[3] == 1
            isSpinner = item[3] == 3

            object = {
                'x':item[0],
                'y':item[1],
                'time':item[2],
                'type':item[3],
                'hitsound':item[4]  # 0 normal, 1 whistle, 2 finish, 3 clap.
            }
            if isSlider:
                object['sliderLength'] = item[7]
            
            if isSpinner:
                object['spinnerEnd'] = item[5]
            
            out['hitobjects'].append(object)

    stars = SR_calculator.calculateSR(file)
    out['sr'] = stars
    assert(out['sr'])

    output = json.dumps(out).replace('\n','')
    
    dest = os.path.join(targetDir, makeSafeFilename(out["metadata"]["Title"].strip()[:20]) + '-' + makeSafeFilename(out["metadata"]["Creator"].strip()[:20]) + '-' + makeSafeFilename(out["metadata"]["Version"].strip()[:20]) + '.json')
   
    with open(dest,'w') as file:
        file.write(output)
    return output