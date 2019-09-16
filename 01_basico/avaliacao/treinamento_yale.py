import cv2
import os
import numpy as np
from PIL import Image #Biblioteca necessária para carregar imagens

eigenface = cv2.face.EigenFaceRecognizer_create(40, 8000)
fisherface = cv2.face.FisherFaceRecognizer_create(3, 2000)
lbph = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 50)

def getImagemComId():
    caminhos = [os.path.join('C:/Users/krupc/Downloads/Github/CursoOpenCV/avaliacao/yalefaces/treinamento', f) for f in os.listdir('C:/Users/krupc/Downloads/Github/CursoOpenCV/avaliacao/yalefaces/treinamento')]
    #caminhos = [os.path.join('treinamento', f) for f in os.listdir('treinamento')]
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        #imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        imagemFace = Image.open(caminhoImagem).convert('L')
        imagemNP = np.array(imagemFace, 'uint8')
        id = int(os.path.split(caminhoImagem)[-1].split('.')[0].replace("subject", ""))
        ids.append(id)
        faces.append(imagemNP)
    return np.array(ids), faces

ids, faces = getImagemComId()

print('Treinando...')
eigenface.train(faces, ids)
eigenface.write('ClassificadorEigenYale.yml')

fisherface.train(faces, ids)
fisherface.write('ClassificadorFisherYale.yml')

lbph.train(faces, ids)
lbph.write('ClassificadorLBPHYale.yml')

print('Treinamento realizado')
