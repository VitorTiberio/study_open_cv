from PIL import Image
import cv2
import numpy as np 
import zipfile 
import os 

path = 'yalefaces.zip'
zip_object = zipfile.ZipFile(path, mode = 'r')
zip_object.extractall('./')
zip_object.close()


def get_image_data():
    arquivos = [os.path.join('yalefaces/train', f) for f in os.listdir('yalefaces/train')]
    faces = []
    ids = []
    for arquivo in arquivos: 
        img = Image.open(arquivo).convert('L')
        img_np = np.array(img, 'uint8')
        id = os.path.split(arquivo)[1].split('.')[0].replace('subject', '')
        ids.append(id)
        faces.append(img_np)
    return np.array(ids), faces

ids, faces = get_image_data()

