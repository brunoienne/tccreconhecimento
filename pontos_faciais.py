import dlib, cv2, numpy as np


def imprimePontos(img, pontos):
    for p in pontos.parts():
        cv2.circle(img, (p.x, p.y), 2, (0, 255, 0), 2)


def imprimeLinhas(img, pontos):
    p68 = [[0, 16, False],
           [17, 21, False],
           [22, 26, False],
           [27, 30, False],
           [30, 35, True],
           [36, 41, True],
           [42, 47, True],
           [48, 59, True],
           [60, 67, True]]
    for k in range(0, len(p68)):
        pts = []
        for i in range(p68[k][0], p68[k][1] + 1):
            ponto = [pontos.part(i).x, pontos.part(i).y]
            pts.append(ponto)
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts], p68[k][2], (255, 0, 0), 2)


fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL
img = cv2.imread("desconhecidos/desconhecido-17-11-2019_16.56.11.jpg")
detector = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
faces = detector(img, 1)
for face in faces:
    pontos = detectorPontos(img, face)
    # imprimePontos(img, pontos)
    imprimeLinhas(img, pontos)
cv2.imshow("Pontos faciais", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
