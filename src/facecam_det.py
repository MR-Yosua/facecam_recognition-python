import cv2


face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    success,img = cap.read()
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img,1.1,8)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w , y+h), (0,0,255), 2)
    
    cv2.imshow('img',img)
    cv2.waitKey(1)
