import cv2

classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)
amostra = 1 #Contador
numeroAmostras = 25 #Quantidade de fotos
id = input('Digite seu identificador: ') 
largura, altura = 200, 200
print('Capturando as faces...')

while(True):
    conectado, imagem = camera.read() #Abre a webcam
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) #Converte imagem para cinza
    facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(100,100))

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2) #Desenha o retângulo na tela
        #k = cv2.waitKey(1)
        #if k%256 == 32: #Space
        if cv2.waitKey(1) & 0xFF == ord('q'):
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura)) #Seleciona só a parte de dentro do quadrinho
            cv2.imwrite("fotos/pessoa-" + str(id) + "-" + str(amostra) + ".jpg", imagemFace) #Salva a imagem
            print("[Foto " + str(amostra) + " capturada com sucesso]")
            amostra += 1

    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    if(amostra >= numeroAmostras + 1):
        break;

print("Fotos capturadas com sucesso!")
camera.release()
cv2.destroyAllWindows()

"""
    As imagens para o treinamento são fundamentais para um reconhecimento eficiente.
    Fazer um ensaio antes de tirar as fotos
    Ambiente bem iluminado
    Variações na expressão (feliz, triste, com e sem óculos)
    Ângulo (olhando levemente para cima, baixo, esquerda, direita)
    Luz incidindo no rosto
"""