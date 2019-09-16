import cv2
import os
import numpy as np
from PIL import Image

detectorFace = cv2.CascadeClassifier("frontal_face.xml")
#reconhecedor= cv2.face.EigenFaceRecognizer_create()
#reconhecedor.read("ClassificadorEigenYale.yml")

#reconhecedor = cv2.face.FisherFaceRecognizer_create()
#reconhecedor.read("ClassificadorFisherYale.yml")

reconhecedor= cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read("ClassificadorLBPHYale.yml")

totalAcertos = 0
percentualAcerto = 0.0
totalConfianca = 0.0

caminhos = [os.path.join('C:/Users/krupc/Downloads/Github/CursoOpenCV/avaliacao/yalefaces/teste', f) for f in os.listdir('C:/Users/krupc/Downloads/Github/CursoOpenCV/avaliacao/yalefaces/teste')]
for caminhoImagem in caminhos:
    imagemFace = Image.open(caminhoImagem).convert('L') #Converte para escala de cinzas
    imagemFaceNP = np.array(imagemFace, 'uint8') #Converte a imagem no formato do numpy array
    facesDetectadas = detectorFace.detectMultiScale(imagemFaceNP)
    for x, y, l, a in facesDetectadas: #Se entrar dentro do for, ele detectou a face na imagem
        '''
        #Teste para ver se ele reconheceu todas as imagens.
        cv2.rectangle(imagemFaceNP, (x, y), (x + l, y + a), (0, 0, 255), 2)
        cv2.imshow("Face", imagemFaceNP)
        cv2.waitKey(1000)
        '''
        idprevisto, confianca = reconhecedor.predict(imagemFaceNP)
        idatual = int(os.path.split(caminhoImagem)[1].split(".")[0].replace("subject", ""))

        if idprevisto == idatual:
            totalAcertos += 1
            totalConfianca += confianca

percentualAcerto = (totalAcertos / 30) * 100
totalConfianca = totalConfianca / totalAcertos

print("Percentual de acertos: " + str(percentualAcerto))
print("Total de confiança: " + str(totalConfianca))

#Exemplo de testes do bacana
#https://bitbucket.org/SpikeSL/vision-systems/src/master/