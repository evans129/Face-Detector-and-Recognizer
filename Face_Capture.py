import os
import cv2
"""
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade=cv2.CascadeClassifier(cascPath)
cap=cv2.VideoCapture(0)
Id=input('enter your id')
sampleNum=0
while(cap.isOpened()):
    success,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imwrite("dataSet/User."+Id+'.'+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        sampleNum=sampleNum+1
        cv2.imshow("output",img)
        if(cv2.waitKey(100)==ord("q")):
            break
        elif(sampleNum>20):
            break
cap.release()
cv2.destroyAllWindows()
"""
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade=cv2.CascadeClassifier(cascPath)
sampleNum=2
img=cv2.imread("/Users/devanshkumar/Downloads/satya5.jpg")
Id="4"
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray,1.1,4)
for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imwrite("dataSet/User."+Id+'.'+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
