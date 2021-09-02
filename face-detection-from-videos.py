# import cv2 library
import cv2

vid = cv2.VideoCapture("videos/faces.mp4")
face_cascade = cv2.CascadeClassifier("haarcascades/frontalface.xml")

# if you want to save the detected video you should use writer variable
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter('detectedfaces.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))

#  examine each frame one by one with an endless loop.
while 1:
    # read each frame one by one.
    _, frame = vid.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 1)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # if you want to save the detected video you should use writer variable
    # writer.write(frame)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
writer.release()
cv2.destroyAllWindows()
