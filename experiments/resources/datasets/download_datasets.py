import zipfile
import gdown
import os

# The datasets are downloaded from the following Google Drive link, you can download them manually if you want.
# https://drive.google.com/drive/folders/1rIkInpp1NBnDytsKOLOFw_Sm5B2Fsxtk?usp=drive_link

datasets = {
    "omniglot": "https://drive.google.com/uc?id=1HGTYfDMPnNrTVLI0wRPL5S2rrMWtZRYW",
    "miniimagenet": "https://drive.google.com/uc?id=1n9yVi-VoVlLs1VGJvQuHekDK9DHtWGIy",
    "fc100": "https://drive.google.com/uc?id=1njMPLaMJ0l21DVSi7X-gBGYZOk_dz20d",
    "cifarfs": "https://drive.google.com/uc?id=1596wnYCH6N-xcOC37apq8LvOg74NeRGI"
}

# Downloading all the datasets into memory.
for key, value in datasets.items():

    print("Downloading " + key.title() + " from Google Drive")
    gdown.download(value, key + ".zip", quiet=False)

    # Unzipping the zip file to extract the folders.
    with zipfile.ZipFile(key + ".zip", 'r') as zip_ref:
        zip_ref.extractall(key)

    # Delete the zip file now that the files are extracted.
    os.remove(key + ".zip")
