import cv2, easygui, time, numpy as np, dlib
from conexao import conexao


class Captura:

    def tiraFotos(self,nome):
        detector = dlib.get_frontal_face_detector()
        cam = cv2.VideoCapture(0)
        ic = 1
        with conexao.Conexao() as db:
            db.execute('insert into usuarios(nome) values(%s)', (nome,))
            id = db.query('select id from usuarios order by id desc limit 1 ')

        print("Usuário " + str(nome) + " cadastrado(a) com sucesso!")
        print("Preparando sessão de fotos em 3s...")
        inicio = time.clock()  # inicia o timer

        while (True):
            conectado, imagem = cam.read()
            faces, confianca, idx = detector.run(imagem)
            segs = (time.clock() - inicio)  # conta os segundos
            for i, face in enumerate(faces):
                e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
                # cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 255), 2)
                if cv2.waitKey(1) and segs > 3 and np.average(imagem) > 110:
                    imgfinal = cv2.resize(imagem, (750, 600))
                    cv2.imwrite("fotos/pessoa." + str(id[0][0]) + "." + str(ic) + ".jpg", imgfinal)
                    print("Foto " + str(ic) + " tirada")
                    ic += 1
                    inicio = time.clock()  # comeca contar novamente
            cv2.imshow("Tela", imagem)
            if cv2.waitKey(1) & ic > 5:
                break
        print("Fotos tiradas com sucesso!")
        cam.release()
        cv2.destroyAllWindows()
