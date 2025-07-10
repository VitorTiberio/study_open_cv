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
    return np.array(ids, dtype=np.int32), faces

ids, faces = get_image_data()

# Treinamento do classificador LBPH #

lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.train(faces, ids)
lbph_classifier.write('lbph_classifier.yml')

# Reconhecendo faces com o classificador LBPH #

lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read('lbph_classifier.yml')

imagem_teste = 'yalefaces/test/subject10.sad.gif'

imagem = Image.open(imagem_teste).convert('L')
imagem_np = np.array(imagem, 'uint8')

previsao = lbph_face_classifier.predict(imagem_np)

cv2.putText(imagem_np, 'Pred: ' +str(previsao[0]), (10,30), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0,255,0))
cv2.imshow('Imagem Encontrada', imagem_np)
cv2.waitKey(0)
cv2.destroyAllWindows()