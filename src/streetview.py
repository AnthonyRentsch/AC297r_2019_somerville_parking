import json
import requests
import re
import shutil

url_img = "https://maps.googleapis.com/maps/api/streetview"

with open('../secret/keys.json') as f:
    KEY = json.load(f)['street_view_key']

def save_streetview_image(location, file_path, width=640, height=640):
    params = {
        'size': f"{width}x{height}",
        'location': location,
        'key': KEY,
        'fov': '120'
    }

    r_image = requests.get(url = url_img, params = params, stream = True)

    with open(file_path, 'wb') as f:
        for chunk in r_image:
            f.write(chunk)
