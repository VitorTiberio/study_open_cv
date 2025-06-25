import cv2
import dlib 

imagem = cv2.imread("tiberio_family_1.jpeg")
imagem = cv2.resize(imagem, (800,600))

face_detector_hog = dlib.get_frontal_face_detector()

detection = face_detector_hog(imagem, 1)

for face in detection: 
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(imagem, (l, t), (r,b), (0,255,0), 2)

cv2.imshow("Rostos detectados", imagem)
cv2.waitKey(0)
