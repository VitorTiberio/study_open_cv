import cv2 

imagem = cv2.imread("foto_01.jpg",1)
imagem_cinza = cv2.imread("foto_01.jpg",0)

detector_facial = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

detection = detector_facial.detectMultiScale(imagem_cinza)

for x,y,w,h in detection: 
    cv2.rectangle(imagem, (x,y),(x + w,y+h),(0,255,0),3)

cv2.imshow("Rostos detectados", imagem)
cv2.waitKey(0)
