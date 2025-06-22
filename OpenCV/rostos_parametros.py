import cv2 

imagem = cv2.imread("tiberio_family_1.jpeg",1)
imagem = cv2.resize(imagem, (800,600))
imagem_cinza = cv2.imread("tiberio_family_1.jpeg",0)
imagem_cinza = cv2.resize(imagem_cinza, (800,600))

detector_facial = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

detection = detector_facial.detectMultiScale(imagem_cinza)

for x,y,w,h in detection: 
    cv2.rectangle(imagem, (x,y),(x + w,y+h),(0,255,0),2)

cv2.imshow("Rostos detectados", imagem)
cv2.waitKey(0)

