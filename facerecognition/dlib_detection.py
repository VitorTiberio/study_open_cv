import cv2
import dlib 

detector_de_face = dlib.get_frontal_face_detector()
detector_pontos = dlib.shape_predictor(r"C:\Users\adm\Documents\IC\Open_CV\face_recognition\Weights\shape_predictor_68_face_landmarks.dat")

imagem = cv2.imread('tiberio_family_1.jpeg')
deteccoes = detector_de_face(imagem,1)

for face in deteccoes: 
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(imagem, (l,t), (r,b), (0,255,0), 2)

cv2.imshow('Rostos Detectados', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()