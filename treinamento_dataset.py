import os, glob, _pickle as cPickle, dlib, cv2, numpy as np, time
from PIL import Image

detector = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
reconhecimento = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")

indice = {}
idx = 0
descritores = None
totalpessoas = 0

time1 = time.time()
for arquivo in glob.glob(os.path.join("yalefaces", "*.gif")):
    imgf = Image.open(arquivo).convert('RGB')
    img = np.array(imgf, 'uint8')
    faces = detector(img, 1)
    numfaces = len(faces)

    if numfaces >= 1:
        totalpessoas += 1

        for face in faces:
            pontosFace = detectorPontos(img, face)
            descritorFace = reconhecimento.compute_face_descriptor(img, pontosFace)
            listaDescritores = [df for df in descritorFace]
            npArrayDescritor = np.asarray(listaDescritores, dtype=np.float64)
            npArrayDescritor = npArrayDescritor[np.newaxis, :]
            if descritores is None:
                descritores = npArrayDescritor
            else:
                descritores = np.concatenate((descritores, npArrayDescritor), axis=0)
            indice[idx] = arquivo
            idx += 1
np.save("descritores_dataset/descritor_dataset.npy", descritores)
with open("descritores_dataset/indices_dataset.pickle", "wb") as f:
    cPickle.dump(indice, f)
time2 = time.time()
print("Classificadores treinados em {:.2f}".format(time2 - time1))
print("Total de faces=" + str(totalpessoas))
