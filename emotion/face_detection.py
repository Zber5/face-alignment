import cv2

# Read the input image
img = cv2.imread('C:/Users/Zber/Desktop/SavedData/Joy_15/left000082.png')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

d = 60
c = d//2

a = 40
b = a //2

# Draw rectangle around the faces and crop the faces
for (x, y, w, h) in faces:
    h = h + d
    w = w + a
    x = x - b
    y = y - c
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    faces = img[y:y + h, x:x + w]
    cv2.imshow("face", faces)
    cv2.imwrite('face1.jpg', faces)

# Display the output
cv2.imwrite('detcted.jpg', img)
cv2.imshow('img', img)
cv2.waitKey()