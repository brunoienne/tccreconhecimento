import os, glob, _pickle as cPickle, dlib, cv2, numpy as np, serial
from PIL import Image

detector = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
reconhecimento = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")
indices = np.load("descritores_dataset/indices_dataset.pickle")
descritoresFaciais = np.load("descritores_dataset/descritor_dataset.npy")
limiar = 0.5
totalfaces = 0
totalacertos = 0
media = 0

for arquivo in glob.glob(os.path.join("yalefaces", "*.gif")):
    imgf = Image.open(arquivo).convert('RGB')
    img = np.array(imgf, 'uint8')
    idatual = int(os.path.split(arquivo)[1].split(".")[0].replace("subject", ""))
    totalfaces += 1
    faces = detector(img, 2)
    for face in faces:
        e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
        pontosFace = detectorPontos(img, face)
        descritorFace = reconhecimento.compute_face_descriptor(img,
                                                               pontosFace)

        listaDecritores = [fd for fd in descritorFace]
        npArrayDescritorFacial = np.asarray(listaDecritores, dtype=np.float64)
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

        distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
        minimo = np.argmin(distancias)
        dMinima = distancias[minimo]
        if dMinima <= limiar:
            nome = os.path.split(indices[minimo])[1].split(".")[0]
            idprevisto = int(os.path.split(indices[minimo])[1].split(".")[0].replace("subject", ""))
            if idprevisto == idatual:
                totalacertos += 1
        else:
            nome = "Desconhecido"
        # print("idatual:{}  idprevisto:{}".format(idatual, idprevisto))
        cv2.rectangle(img, (e, t), (d, b), (0, 255, 0), 2)
        texto = "{} {:.4f}".format(nome, dMinima)
        cv2.putText(img, texto, (d, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255))

    cv2.imshow("Tela", img)
    cv2.waitKey(0)

percentualacerto = (totalacertos / totalfaces) * 100
print("Percentual de acertos:{0:.2f}".format(percentualacerto))
cv2.destroyAllWindows()
