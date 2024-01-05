import cv2
import numpy as np
import face_recognition
import os

path = 'attendence'
image = []
className = []
myList = os.listdir(path)
for cls in myList:
    curImage = cv2.imread(f'{path}/{cls}')
    image.append(curImage)
    className.append(os.path.splitext(cls)[0])

def findEncoding(image):
    encodelist = []
    for im in image:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(im)[0]
        encodelist.append(encode)
    return encodelist

EncodeListKnow = findEncoding(image)
cap = cv2.VideoCapture(0)
frame_number = 0
tm = cv2.TickMeter()
while True:
    success, img = cap.read()
    imgs = cv2.resize(img, (0,0),None, 0.25, 0.25)
    imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    tm.start()
    face_location = face_recognition.face_locations(imgs)
    face_encoding = face_recognition.face_encodings(imgs, face_location)
    if len(face_location) >0 and frame_number % 3 == 0:
        for face_loca, face_en in zip(face_location, face_encoding):
            matches = face_recognition.compare_faces(EncodeListKnow, face_en)
            face_dis = face_recognition.face_distance(EncodeListKnow, face_en)
            if face_dis[0] < 0.5:
                idx = np.argmin(face_dis)
                name = className[idx]
            else: 
                name = "unknow"
            y1, x2, y2, x1 = face_loca
            cv2.rectangle(img, (x1, y1), (x2, y2), (225,225,0), 2)
            cv2.putText(img, name, (x1+6,y2+15), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    tm.stop()
    frame_number += 1
    cv2.putText(img, 'Fps: {:.2f}'.format(tm.getFPS()), (0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
    tm.reset()
    cv2.imshow('camera', img)
    cv2.waitKey(1)