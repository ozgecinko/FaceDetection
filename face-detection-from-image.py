# import cv2 library
import cv2

# read the image
img = cv2.imread("images/harrypotter.jpg")

# add the cascade file for face detection
face_cascade = cv2.CascadeClassifier("haarcascades/frontalface.xml")

# turn the haar-like properties to grayscale to detect well
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find the faces coordinates using cascade
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# get the faces inside the rectangle using the coordinates we keep in the faces variable.
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 69, 255), 2)

cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
