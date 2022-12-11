import streamlit as st
import cv2
import os
from PIL import Image
import  numpy as np
caspath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
face_cascade=cv2.CascadeClassifier(caspath)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("trainingData.yml")
def detect_faces(image):
    img=np.array(image.convert('RGB'))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,4)
    name="unknown"
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        id,uncertainty=rec.predict(gray[y:y+h,x:x+w])
        print(id,uncertainty)
        if(uncertainty<63):
            if(id==4):
                name = "SATYA NADELLA"
                cv2.putText(img, name, (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0, 0, 255))
        else:
            cv2.putText(img,'Unknown', (x, y + h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (0, 0, 255))
    return img
def main():
    st.title("Facerecognition")
    html_temp="""
    <body style="background-color:red;">
    <div style="background-color:teal;padding:10px">
    <h2 style="color:white;text-align:center;">Face recognition app</h2>
    </div>
    </body>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    image_file=st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    if image_file is not None:
        image=Image.open(image_file)
        st.text("original Image")
        st.image(image)
    if st.button("Recognise"):
        result_img=detect_faces(image)
        st.image(result_img)
if __name__== '__main__':
    main()


