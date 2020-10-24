import os, glob, _pickle as cPickle, dlib, cv2, numpy as np
from os import path

detector = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
reconhecimento = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")
indices = np.load("descritores\indices_pessoa.pickle")

indice = {}
idx = 0
descritores = None

print(indices)

for arquivo in glob.glob(os.path.join("fotos", "*")):
    print(arquivo)
    img = cv2.imread(arquivo)
    print(np.average(img))
    # faces = detector(img, 1)

#     for face in faces:
#         print("Achou face")
#         pontosFace = detectorPontos(img, face)  # obtem os 68 pontos
#         descritorFace = reconhecimento.compute_face_descriptor(img, pontosFace)  # retorna vetor de 128
#         listaDescritores = [df for df in descritorFace]  # constroi uma lista
#         npArrayDescritor = np.asarray(listaDescritores, dtype=np.float64)
#         npArrayDescritor = npArrayDescritor[np.newaxis, :]  # dimensiona a lista 1x128
#         if descritores is None:
#             descritores = npArrayDescritor
#         else:
#             descritores = np.concatenate((descritores, npArrayDescritor), axis=0)
#         indice[idx] = arquivo
#         idx += 1
# print(indice)
# np.save("descritores/descritor_pessoa.npy", descritores)  # [idx - arquivo.id.indice]
# with open("descritores/indices_pessoa.pickle", "wb") as f:
#     cPickle.dump(indice, f)  # grava o indice(nome img) no arquivo f
