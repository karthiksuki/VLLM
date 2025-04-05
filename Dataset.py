import os
import zipfile
import requests
from tqdm import tqdm

def download_and_extract(url, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    filename = url.split("/")[-1]
    zip_path = os.path.join(dest_folder, filename)

    with requests.get(url, stream=True) as r, open(zip_path, 'wb') as f:
        total = int(r.headers.get('content-length', 0))
        for data in tqdm(r.iter_content(chunk_size=1024), total=total//1024, unit='KB'):
            f.write(data)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    os.remove(zip_path)

download_and_extract("http://images.cocodataset.org/zips/train2017.zip", "./coco/images")
download_and_extract("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", "./coco/annotations")
