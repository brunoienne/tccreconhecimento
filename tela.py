import tkinter as tk, cv2, _pickle as cPickle, dlib, numpy as np, datetime as dt, serial, os, glob, time
from PIL import Image, ImageTk
from conexao import conexao
from tkinter import filedialog


class Tela:
    def __init__(self, janela):
        # Config janela
        self.janela = janela
        self.janela.title("Reconhecimento Facial Aplicado no Controle de Acesso Residencial")
        self.janela.config(background="#FFFFFF")
        self.frame = None
        self.counter = 0
        self.id = None
        self.detectado = False
        self.lb = None

        # Variaveis
        self.reconhecendo = False
        self.limiar = 0.45
        self.fotos = 3
        self.nomes = None
        self.reconhecido = 0
        self.desconhecido = 0
        self.pos = 0
        self.tam = 0
        self.imagens = []
        self.ser = serial.Serial('COM3', 9600)

        # Tenta abrir a webcam por padrão
        self.cam = cv2.VideoCapture(0)
        self.detector = dlib.get_frontal_face_detector()

        # Arquivos descritores
        self.detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
        self.reconhecimento = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")
        self.subdetector = ["Frente", "Esquerda", "Direita", "Girando esquerda", "Girando direita"]
        self.indices = None
        self.descritoresFaciais = None
        self.select_users()

        ############################# Lado esquerdo #############################
        self.quadro = tk.Frame(self.janela, width=600, height=500)
        self.quadro.grid(row=0, column=0, padx=10, pady=2, rowspan=10)

        self.painel = tk.Label(self.quadro)
        self.painel.grid(row=0, column=0)

        self.dlog = tk.Label(self.janela, text="Log aplicação", font="Consolas 11 bold")
        self.dlog.grid(row=10, column=0, padx=10, pady=2)

        self.caixaLog = tk.Text(self.janela, borderwidth=3, height=8)
        self.caixaLog.grid(row=11, column=0, padx=10, pady=2)

        self.barrRol = tk.Scrollbar(self.janela, command=self.caixaLog.yview)
        self.barrRol.grid(row=11, column=0, sticky="nse", padx=10)
        self.caixaLog['yscrollcommand'] = self.barrRol.set

        ############################# Lado direito #############################

        # Frame menu(maior)
        self.opcoes = tk.LabelFrame(self.janela, text="Menu", font="Consolas 11 bold")
        self.opcoes.grid(row=0, column=1, rowspan=24, padx=5, pady=10, sticky='nsew')

        self.branco = tk.Label(self.opcoes)
        self.branco.grid(row=1, column=1, sticky='we')

        # Cadastrar frame
        self.cad = tk.LabelFrame(self.opcoes, text="Cadastrar", font="Consolas 11 bold")
        self.cad.grid(row=2, column=1, sticky='nsew', padx=5)

        self.lnome = tk.Label(self.cad, text="Nome:", font="Consolas 11")
        self.lnome.grid(row=3, column=1, sticky='w')

        self.sval = tk.StringVar()
        self.nome = tk.Entry(self.cad, textvariable=self.sval)  # input de texto
        self.nome.grid(row=4, column=1, padx=30, sticky='w')

        self.botao = tk.Button(self.cad, text="Salvar", command=self.salva_nome)
        self.botao.grid(row=5, column=1, pady=5)

        self.branco = tk.Label(self.opcoes)
        self.branco.grid(row=7, column=1, sticky='we')

        # Frame reconhecer
        self.rec = tk.LabelFrame(self.opcoes, text="Reconhecer", font="Consolas 11 bold")
        self.rec.grid(row=12, column=1, sticky='nsew', padx=5)

        self.btn_rec = tk.Button(self.rec, text="Reconhecer", command=self.reconhecer)
        self.btn_rec.grid(row=13, column=1, padx=60, pady=5)

        self.btn_prec = tk.Button(self.rec, text="Parar", command=self.parar, state="disabled")
        self.btn_prec.grid(row=14, column=1, pady=5)

        self.branco = tk.Label(self.opcoes)
        self.branco.grid(row=15, column=1, sticky='we')

        # Frame salvar log
        self.log = tk.LabelFrame(self.opcoes, text="Log", font="Consolas 11 bold")
        self.log.grid(row=16, column=1, sticky='nsew', padx=5)

        self.btn_slog = tk.Button(self.log, text="Salvar log", state="disabled", command=self.salva_log)
        self.btn_slog.grid(row=17, column=1, padx=60, pady=5)

        self.branco = tk.Label(self.opcoes)
        self.branco.grid(row=18, column=1, sticky='we')

        # Frame excluir usuarios
        self.del_user = tk.LabelFrame(self.opcoes, text="Usuarios", font="Consolas 11 bold")
        self.del_user.grid(row=19, column=1, sticky='nsew', padx=5)

        self.btn_duser = tk.Button(self.del_user, text="Cadastrados", command=self.exibe_users)
        self.btn_duser.grid(row=20, column=1, padx=50, pady=5)

        self.branco = tk.Label(self.opcoes)
        self.branco.grid(row=21, column=1, sticky='we')

        self.btn_desc = tk.Button(self.del_user, text="Desconhecidos", command=self.exibe_desconhecidos)
        self.btn_desc.grid(row=22, column=1, padx=50, pady=5)

        self.branco = tk.Label(self.opcoes)
        self.branco.grid(row=23, column=1, sticky='we')

        # Frame destravar porta
        self.porta = tk.LabelFrame(self.opcoes, text="Porta", font="Consolas 11 bold")
        self.porta.grid(row=24, column=1, sticky='nsew', padx=5)

        self.btn_unlock = tk.Button(self.porta, text="Destravar", command=self.unlock)
        self.btn_unlock.grid(row=25, column=1, padx=70, pady=5)

        self.branco = tk.Label(self.opcoes)
        self.branco.grid(row=26, column=1, sticky='we')

        self.btn_sair = tk.Button(self.opcoes, text="Sair", command=self.sair)
        self.btn_sair.grid(row=27, column=1)

        # Uma vez que o update é chamado o frame é atualizado a cada 15ms
        self.delay = 15
        self.update()
        self.janela.mainloop()

    def update(self):
        if self.reconhecendo != True:
            # Captura frame da cam
            ret, frame = self.cam.read()
            if ret:
                self.frame = frame.copy()
                faces, confianca, idx = self.detector.run(frame)
                if len(faces) == 1:
                    self.detectado = True
                else:
                    self.detectado = False
                for i, face in enumerate(faces):
                    e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
                    cv2.rectangle(frame, (e, t), (d, b), (0, 255, 255), 2)
                    # self.insere_msg("Val. de luminosidade= {0:.2f}".format(np.average(self.frame)))
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                image = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=image)
                self.painel.imgtk = imgtk
                self.painel.config(image=imgtk)
                self.janela.after(self.delay, self.update)

    def salva_nome(self):
        if self.sval.get() != '':
            with conexao.Conexao() as db:
                db.execute('insert into usuarios(nome) values(%s)', (self.sval.get(),))
                self.id = db.query('select id from usuarios order by id desc limit 1 ')
            self.insere_msg("Usuário " + str(self.sval.get()) + " inserido(a) com sucesso!")
            self.nome.delete(0, tk.END)
            self.nome.config(state="disabled")
            self.botao.config(state="disabled")
            self.btn_slog.config(state="normal")
            # self.insere_msg("Iniciando sessão de " + self.fotos + " fotos em 3 segundos...")
            self.counter = 1
            self.janela.after(3000, self.captura_foto)
        else:
            self.insere_msg("Insira um nome para cadastro.")

    def captura_foto(self):
        if self.counter != 0:
            if np.average(self.frame) >= 110:
                if self.detectado:
                    if self.counter <= self.fotos:
                        imgfinal = cv2.resize(self.frame, (320, 180))
                        cv2.imwrite("fotos/pessoa." + str(self.id[0][0]) + "." + str(self.counter) + ".jpg", imgfinal)
                        self.insere_msg("Foto " + str(self.counter) + " tirada!")

                        if self.counter == self.fotos:
                            self.counter = 0
                            self.insere_msg("Capturas realizadas com sucesso!")
                            self.insere_msg("Iniciando treinamento dos classificadores...")
                            self.inicia_treinar()
                        else:
                            self.counter += 1
                            self.janela.after(3000, self.captura_foto)
                else:
                    self.insere_msg("Rosto deve estar sendo detectado!")
                    self.janela.after(3000, self.captura_foto)
            else:
                self.insere_msg("Melhore a iluminação da imagem!")
                print(np.average(self.frame))
                self.janela.after(3000, self.captura_foto)

    def inicia_treinar(self):
        self.janela.after(2000, self.treinar)

    def treinar(self):
        idx = 0
        indice = {}
        descritores = None
        time1 = time.time()
        for arquivo in glob.glob(os.path.join("fotos", "*.jpg")):
            img = cv2.imread(arquivo)
            faces = self.detector(img, 1)
            if len(faces) != 1:
                self.insere_msg("Nenhuma face encontrada no arquivo {}".format(arquivo))
            else:
                for face in faces:
                    pontosFace = self.detectorPontos(img, face)  # obtem os 68 pontos
                    descritorFace = self.reconhecimento.compute_face_descriptor(img, pontosFace)  # retorna vetor de 128
                    listaDescritores = [df for df in descritorFace]  # constroi uma lista
                    npArrayDescritor = np.asarray(listaDescritores, dtype=np.float64)
                    npArrayDescritor = npArrayDescritor[np.newaxis, :]  # dimensiona a lista 1x128
                    if descritores is None:
                        descritores = npArrayDescritor
                    else:
                        descritores = np.concatenate((descritores, npArrayDescritor), axis=0)

                    indice[idx] = arquivo
                    idx += 1
        np.save("descritores/descritor_pessoa.npy", descritores)  # [idx,vetor de 128 posicoes]
        with open("descritores/indices_pessoa.pickle", "wb") as f:
            cPickle.dump(indice, f)  # [idx - arquivo.id.indice]

        time2 = time.time()
        self.nome.config(state="normal")
        self.botao.config(state="normal")
        self.btn_slog.config(state="normal")
        self.insere_msg("Classificadores treinados em {:.2f} segundos!".format(time2 - time1))

    def reconhecer(self):
        if os.path.exists("descritores/indices_pessoa.pickle"):
            self.insere_msg("Iniciado reconhecimento.")
            self.btn_rec.config(state="disabled")
            self.btn_prec.config(state="normal")
            self.btn_slog.config(state="normal")
            self.load_files()
            self.select_users()
            self.reconhecendo = True
            self.identificando()
        else:
            self.insere_msg("Nenhum usuário cadastrado.")

    def identificando(self):
        if self.reconhecendo:
            ret, frame = self.cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces, confianca, idx = self.detector.run(gray)

            if len(faces) > 0:  # se encontrar alguma face ele entra no for
                for i, face in enumerate(faces):
                    e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
                    pontosFace = self.detectorPontos(frame, face)
                    descritorFace = self.reconhecimento.compute_face_descriptor(frame,
                                                                                pontosFace)  # caracteristicas face sendo gravada
                    listaDescritores = [fd for fd in descritorFace]
                    npArrayDescritorFacial = np.asarray(listaDescritores, dtype=np.float64)
                    npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

                    # calculo da distancia euclidiana da face atual com as das imgs gravadas
                    distancias = np.linalg.norm(npArrayDescritorFacial - self.descritoresFaciais, axis=1)
                    minimo = np.argmin(distancias)  # pega a menor distancia (indice)
                    dMinima = distancias[minimo]

                    if dMinima <= self.limiar:
                        self.reconhecido += 1
                        self.desconhecido = 0
                        idpessoa = os.path.split(self.indices[minimo])[1].split(".")[1]
                        nomePessoa = str([i[1] for i in self.nomes if i[0] == int(idpessoa)]).strip("['']")

                        if self.reconhecido == 4:
                            self.parar()
                            self.ser.write(b'1')
                            self.insere_msg(
                                "Bem vindo " + nomePessoa + ", reconhecido(a) com distância igual à {0:.4f}".format(
                                    dMinima).replace('.', ','))

                    else:
                        self.desconhecido += 1
                        self.reconhecido = 0
                        nomePessoa = "Desconhecido(a)"

                        if self.desconhecido == 4:
                            self.parar()
                            self.ser.write(b'0')
                            img = cv2.resize(frame, (640, 360))
                            cv2.imwrite("desconhecidos/desconhecido-" + str(dt.datetime.now().strftime(
                                '%d-%m-%Y_%H.%M.%S')) + ".jpg", img)
                            self.insere_msg("Usuário desconhecido.")

                    texto = "{} {:.4f}".format(nomePessoa, dMinima).replace('.', ',')
                    cv2.putText(frame, texto, (d, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
                    cv2.rectangle(frame, (e, t), (d, b), (0, 255, 0), 2)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=image)
            self.painel.imgtk = imgtk
            self.painel.config(image=imgtk)
            self.janela.after(self.delay, self.identificando)

    def parar(self):
        self.btn_rec.config(state="normal")
        self.btn_prec.config(state="disabled")
        self.reconhecendo = False
        self.reconhecido = 0
        self.desconhecido = 0
        self.update()

    def salva_log(self):
        f = filedialog.asksaveasfile(mode='w', defaultextension='.txt', filetypes=[('Texto', '.txt')])
        if f is not None:
            f.write(self.caixaLog.get("1.0", tk.END))
            f.close()

    def select_users(self):
        with conexao.Conexao() as db:
            self.nomes = db.query('select id,nome from usuarios order by nome')  # vetor de nomes cadastrados

    def exibe_users(self):
        self.select_users()
        tel2 = tk.Toplevel(self.janela)
        tel2.title("Usuários cadastrados")
        tel2.geometry("%dx%d%+d%+d" % (300, 230, 350, 200))
        tel2.resizable(False, False)
        self.lb = tk.Listbox(tel2)
        self.lb.grid(row=0, column=0, padx=(90, 0), pady=10)
        for nm in self.nomes:
            self.lb.insert(tk.END, nm[1])
        self.lb.select_set(0)
        del_user = tk.Button(tel2, text="Excluir", command=self.del_usuario)
        del_user.grid(row=1, column=0, padx=(90, 0))
        if not self.nomes:
            del_user.config(state="disabled")
        else:
            del_user.config(state="normal")
        sb = tk.Scrollbar(tel2, command=self.lb.yview)
        sb.grid(row=0, column=0, sticky="nse", pady=10)
        self.lb['yscrollcommand'] = sb.set

    def exibe_desconhecidos(self):
        tel3 = tk.Toplevel(self.janela)
        tel3.title("Usuários desconhecidos")
        tel3.geometry("%dx%d%+d%+d" % (740, 460, 250, 100))
        tel3.resizable(False, False)
        for image in glob.glob("desconhecidos/*.jpg"):
            self.imagens.append(image)
        self.tam = len(self.imagens)
        if self.tam > 0:
            path = self.imagens[self.pos]
            pathf = path.replace("desconhecidos\\", "")
            img = ImageTk.PhotoImage(Image.open(path))
            lab1 = tk.Label(tel3, image=img)
            lab1.image = img
            lab2 = tk.Label(tel3, text=pathf, font="Consolas 10 bold")
            btn_dir = tk.Button(tel3, text="→", width=5, height=5, command=lambda: self.loop_imgs('D', lab1, lab2))
            btn_esq = tk.Button(tel3, text="←", width=5, height=5, command=lambda: self.loop_imgs('E', lab1, lab2))
            lab1.pack()
            lab2.pack()
            btn_dir.pack(side=tk.RIGHT)
            btn_esq.pack(side=tk.LEFT)

    def load_files(self):
        self.indices = np.load("descritores/indices_pessoa.pickle")
        self.descritoresFaciais = np.load("descritores/descritor_pessoa.npy")

    def delete_files(self):
        os.remove("descritores/descritor_pessoa.npy")
        os.remove("descritores/indices_pessoa.pickle")

    def insere_msg(self, msg):
        self.caixaLog.insert(tk.INSERT, dt.datetime.now().strftime(
            '%H:%M:%S') + " - " + msg + "\n")

    def del_usuario(self):
        # indexes = []
        self.load_files()
        nomeUser = self.lb.get(self.lb.curselection())
        idUser = str([i[0] for i in self.nomes if i[1] == nomeUser]).strip('[]')
        self.select_users()
        with conexao.Conexao() as db:
            db.execute('delete from usuarios where id = %s', (idUser,))
        self.lb.delete(self.lb.curselection())

        # remove as fotos do usuario
        for file in glob.glob("fotos/pessoa." + idUser + "*"):
            os.remove(file)

        self.delete_files()

        if len(self.nomes) > 1:
            self.treinar()

        # if len(self.nomes) == 1:
        #     self.delete_files()
        #
        # else:
        #     # remove vetor do usuario de descritores
        #     for i, value in self.indices.items():
        #         if "pessoa." + idUser + "" in value:
        #             indexes.append(i)
        #     x = np.delete(self.descritoresFaciais, indexes[0], axis=0)
        #     np.save("descritores/descritor_pessoa.npy", x)
        #
        #     # remove indice do arquivo de indices
        #     for ix in indexes:
        #         del self.indices[ix]
        #
        #     with open("descritores/indices_pessoa.pickle", "wb") as f:
        #         cPickle.dump(self.indices, f)

        self.insere_msg("Usuario " + nomeUser + " excluído com sucesso!")

    def unlock(self):
        print('')
        self.ser.write(b'1')

    def lock(self):
        print('')
        self.ser.write(b'0')

    def loop_imgs(self, drc, lb1, lb2):
        if drc == 'D' and self.pos != self.tam - 1:
            self.pos += 1
        elif drc == 'E' and self.pos != 0:
            self.pos -= 1

        path = self.imagens[self.pos]
        pathf = path.replace("desconhecidos\\", '')
        img = ImageTk.PhotoImage(Image.open(path))
        lb1.configure(image=img)
        lb1.image = img
        lb2.configure(text=pathf)

    def sair(self):
        self.lock()
        self.janela.destroy()


def main():
    # Cria a janela
    tel = Tela(tk.Tk())


if __name__ == "__main__":
    main()
