import numpy as np
import face_recognition
import cv2
import time 
start = time.time()
elon = face_recognition.load_image_file('image/Huy.jpg')
elon = cv2.cvtColor(elon, cv2.COLOR_BGR2RGB)
tri = face_recognition.load_image_file('image/huy2.jpg')
tri = cv2.cvtColor(tri, cv2.COLOR_BGR2RGB)


face_loca = face_recognition.face_locations(elon)[0]
face_encode = face_recognition.face_encodings(elon)[0]
cv2.rectangle(elon, (face_loca[3], face_loca[0]), (face_loca[1], face_loca[2]), (225,225,0), 2)

face_loca_tri = face_recognition.face_locations(tri)[0]
face_encode_tri = face_recognition.face_encodings(tri)[0]
cv2.rectangle(tri, (face_loca_tri[3], face_loca_tri[0]), (face_loca_tri[1], face_loca_tri[2]), (225,225,0), 2)

result = face_recognition.compare_faces([face_encode], face_encode_tri)
face_dis = face_recognition.face_distance([face_encode], face_encode_tri)
print(result, face_dis)
end = time.time()
runtime = end - start
print(runtime)
# cv2.imshow('tri', tri)
# cv2.waitKey(0)
# cv2.imshow('elon', elon)
# cv2.waitKey(0)