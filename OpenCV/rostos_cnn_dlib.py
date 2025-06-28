import cv2
import dlib 

imagem = cv2.imread("foto_01.jpg")

detector_face = dlib.cnn_face_detection_model_v1("Weights\mmod_human_face_detector.dat")

detection = detector_face(imagem, 2)

for face in detection: 
    l, t, r, b, c = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
    cv2.rectangle(imagem, (l,t), (r,b), (0,255,0), 2)

cv2.imshow("Rostos detectados", imagem)
cv2.waitKey(0)