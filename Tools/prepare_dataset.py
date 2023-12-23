import os
import requests
import zipfile


def download_and_unzip(database_name):
    if database_name == 'D1':
        url = 'https://drive.google.com/file/d/1Dn5jv8PNX-NHKdkGJ9velyG_4AI--Cz9/view?usp=drive_link'
    elif database_name == 'D2':
        url = 'https://drive.google.com/file/d/16VDeLf5lXPz7qkhhddziXXJmjQ2Nyp22/view?usp=drive_link'
    else:
        raise RuntimeError("Database name is incorrect. It should be D1 or D2.")

    destination = '../data/'

    zip_path = os.path.join(destination, database_name + ".zip")

    if not os.path.exists(destination):
        os.makedirs(destination)

    # Check if the zip file is already downloaded
    if not os.path.exists(zip_path):
        file_id = url.split("/")[-2]
        direct_download_link = f'https://drive.google.com/uc?id={file_id}'
        response = requests.get(direct_download_link)
        if "Google Drive can't scan this file for viruses" in response.text:
            confirm_param = response.text.split('confirm=')[1].split('method')[0]
            direct_download_link = f'https://drive.google.com/uc?id={file_id}&confirm={confirm_param}'
            response = requests.get(direct_download_link, stream=True, allow_redirects=True)

        with open(zip_path, "wb") as zip_file:
            zip_file.write(response.content)

    # Check if the destination folder is already unzipped
    if not os.path.exists(os.path.join(destination, database_name)):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination)



