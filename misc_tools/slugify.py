import json, os, unicodedata, re

def slugify(name):  # taken from Django's slugify function
    # Normalizes string, converts to lowercase, removes non-alpha characters, and converts spaces to underscores.
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    name = re.sub(r'[^\w\s-]', '', name.lower())
    return re.sub(r'[-\s]+', '_', name).strip('-_')