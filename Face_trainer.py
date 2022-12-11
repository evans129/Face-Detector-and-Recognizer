import cv2
import os
import numpy as np
from PIL import Image

recognizer=cv2.face.LBPHFaceRecognizer_create()
path='dataset'
def getImageWidthId(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagepath in imagePaths:
        faceImg=Image.open(imagepath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        print(imagepath)
        ID=int(os.path.split(imagepath)[-1].split(".")[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return np.array(IDs),faces
Ids,faces=getImageWidthId(path)
recognizer.train(faces,Ids)
recognizer.save('trainingData.yml')
cv2.destroyAllWindows()