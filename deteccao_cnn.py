import cv2, dlib

img = cv2.imread("fotos/grupo.0.jpg")
detector = dlib.cnn_face_detection_model_v1("recursos/mmod_human_face_detector.dat")
faces = detector(img, 1)
print("Faces detectadas:", len(faces))

for face in faces:
    e, t, d, b, co = (int(face.rect.left()), int(face.rect.top()), int(face.rect.right()), int(face.rect.bottom()),
                     face.confidence)
    print(co)
    cv2.rectangle(img, (e, t), (d, b), (255, 255, 0), 2)
cv2.imshow("Detector CNN", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
