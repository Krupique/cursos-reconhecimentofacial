import cv2
import sys

#cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)
largura, altura = 200, 200
amostra = 1

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if cv2.waitKey(1) & 0xFF == ord('w'):
            imagemFace = cv2.resize(gray[y:y + w, x:x + h], (largura, altura))
            cv2.imwrite("fotos/pessoa-" + str(id) + "-" + str(amostra) + ".jpg", imagemFace)
            print("[Foto " + str(amostra) + " capturada com sucesso]")
            amostra += 1


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()