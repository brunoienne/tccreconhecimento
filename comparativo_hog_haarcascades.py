import os, glob, _pickle as cPickle, dlib, cv2, numpy as np, time
from PIL import Image

detector = dlib.get_frontal_face_detector()
classificador = cv2.CascadeClassifier("recursos/haarcascade-frontalface-default.xml")

dtc0 = 0
dtc1 = 0
dtc2 = 0
dtc3 = 0

for arquivo in glob.glob(os.path.join("fotos_dataset", "*.jpg")):
    imgf = Image.open(arquivo)
    img = np.array(imgf, 'uint8')
    imgc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # facdtc = classificador.detectMultiScale(imgc)
    facdtc = detector(img)
    numfaces = len(facdtc)
    # for (x, y, l, a) in facdtc:
    #     cv2.rectangle(img, (x, y), (x + l, y + a), (0, 255, 0), 2)
    # cv2.imshow("Titulo", img)
    # cv2.waitKey(0)
    if numfaces == 0:
        dtc0 += 1
    elif numfaces == 1:
        dtc1 += 1
    elif numfaces == 2:
        dtc2 += 1
    elif numfaces == 3:
        dtc3 += 1
print("Nenhuma face=" + str(dtc0))
print("Unica face=" + str(dtc1))
print("Duas faces=" + str(dtc2))
print("Três faces=" + str(dtc3))
