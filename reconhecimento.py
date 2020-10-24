import os, dlib, cv2, numpy as np, serial
from conexao import conexao

# ser = serial.Serial('COM3', 9600)
cam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
reconhecimento = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")
indices = np.load("descritores/indices_pessoa.pickle")
descritoresFaciais = np.load("descritores/descritor_pessoa.npy")
limiar = 0.5
reconhecido = 0
desconhecido = 0

with conexao.Conexao() as db:
    nomes = db.query('select id,nome from usuarios order by id')  # vetor de nomes cadastrados

while (True):
    conectado, imagem = cam.read()
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    faces, confianca, idx = detector.run(gray)
    if len(faces) > 0:  # se encontrar alguma face ele entra no for
        for i, face in enumerate(faces):
            e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
            pontosFace = detectorPontos(imagem, face)
            descritorFace = reconhecimento.compute_face_descriptor(imagem,
                                                                   pontosFace)  # caracteristicas face sendo gravada

            listaDecritores = [fd for fd in descritorFace]
            npArrayDescritorFacial = np.asarray(listaDecritores, dtype=np.float64)
            npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

            # calculo da distancia euclidiana da face atual com as das imgs gravadas
            distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
            minimo = np.argmin(distancias)  # pega a menor distancia
            dMinima = distancias[minimo]
            if dMinima <= limiar:
                reconhecido += 1
                desconhecido = 0
                idpessoa = os.path.split(indices[minimo])[1].split(".")[1]
                nomePessoa = [i[1] for i in nomes if i[0] == int(idpessoa)]
                # if reconhecido == 8:
                    # ser.write(b'1')
            else:
                desconhecido += 1
                reconhecido = 0
                nomePessoa = "Desconhecido(a)"
                # if desconhecido == 8:
                    # ser.write(b'0')

            texto = "{} {:.4f}".format(nomePessoa, dMinima)
            cv2.putText(imagem, texto, (d, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
            cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 0), 2)

    # print(str(reconhecido))
    cv2.imshow("Tela", imagem)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
