import cv2
import dlib 
import os
import numpy as np 
from PIL import Image

detector_de_face = dlib.get_frontal_face_detector()
detector_pontos = dlib.shape_predictor('Weights/shape_predictor_68_face_landmarks.dat')
descritor_facial_extrator = dlib.face_recognition_model_v1('Weights\dlib_face_recognition_resnet_model_v1.dat')

index = {}
idx = 0
descritores_faciais = None

paths = [os.path.join('face_recognition/yalefaces/train', f) for f in os.listdir('face_recognition/yalefaces/train')]
for path in paths: 
    imagem = Image.open(path).convert('RGB')
    imagem_np = np.array(imagem, 'uint8')
    deteccoes = detector_de_face(imagem_np, 1)
    for face in deteccoes: 
        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(imagem_np, (l,t), (r,b), (0,255,255), 2)
        pontos = detector_pontos(imagem_np, face)
        for ponto in pontos.parts(): 
            cv2.circle(imagem_np, (ponto.x, ponto.y), 2, (0,255,255), 1)
        descritor_facial = descritor_facial_extrator.compute_face_descriptor(imagem_np, pontos)
        descritor_facial = [f for f in descritor_facial]
        descritor_facial = np.asarray(descritor_facial, dtype = np.float64)
        descritor_facial = descritor_facial[np.newaxis, :]

    if descritores_faciais == None: 
        descritores_faciais = descritor_facial
    else: 
        descritores_faciais = np.concatenate((descritores_faciais, descritor_facial), axis = 0)
    index[idx] = path 
    idx = idx + 1

cv2.imshow('Rostos Detectados', imagem_np)
cv2.waitKey(0)