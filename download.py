import requests
import os
from tqdm import tqdm

ID = '1NLsYmwVaRLLCdxKmteSPVUc9EcFCToeG'
DESTINATION = 'tmp/'


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


if not os.path.exists(DESTINATION):
    os.makedirs(DESTINATION)

download_file_from_google_drive(ID, DESTINATION + 'data.zip')

import zipfile

zip_ref = zipfile.ZipFile(DESTINATION + 'data.zip', 'r')
zip_ref.extractall('assets/')
zip_ref.close()