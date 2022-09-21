import cv2
from cv2 import CascadeClassifier
import numpy as np
from pickletools import read_bytes8
from flask import Flask,render_template
app = Flask(__name__)
@app.route('/')
def test():
    return render_template('seatbelt.html')


@app.route('/project')
def pro():
    ans=""

    #Slope of line
    def Slope(a,b,c,d):
        return (d - b)/(c - a)

    # Reading Image
    #beltframe = cv2.imread("test.jpg")
    beltframe = cv2.imread("maxresdefault.jpg")

    #Converting To GrayScale
    beltgray = cv2.cvtColor(beltframe,cv2.COLOR_BGR2GRAY)

    # No Belt Detected Yet
    belt = False

    # Bluring The Image For Smoothness
    blur = cv2.blur(beltgray, (1, 1))

    # Converting Image To Edges
    edges = cv2.Canny(blur, 50, 400)

    # Previous Line Slope
    ps = 0

    # Previous Line Co-ordinates
    px1, py1, px2, py2 = 0, 0, 0, 0

    # Extracting Lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/270, 30, maxLineGap = 20, minLineLength = 170)

    # If "lines" Is Not Empty
    if lines is not None:

        # Loop line by line
        for line in lines:

            # Co-ordinates Of Current Line
            x1, y1, x2, y2 = line[0]

            # Slope Of Current Line
            s = Slope(x1,y1,x2,y2)
            
            # If Current Line's Slope Is Greater Than 0.7 And Less Than 2
            if ((abs(s) > 0.7) and (abs (s) < 2)):

                # And Previous Line's Slope Is Within 0.7 To 2
                if((abs(ps) > 0.7) and (abs(ps) < 2)):

                    # And Both The Lines Are Not Too Far From Each Other
                    if(((abs(x1 - px1) > 5) and (abs(x2 - px2) > 5)) or ((abs(y1 - py1) > 5) and (abs(y2 - py2) > 5))):

                        # Plot The Lines On "beltframe"
                        cv2.line(beltframe, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.line(beltframe, (px1, py1), (px2, py2), (0, 0, 255), 3)

                        # Belt Is Detected
                        print ("Belt Detected")
                        ans="Belt Detected"
                        belt = True

            # Otherwise Current Slope Becomes Previous Slope (ps) And Current Line Becomes Previous Line (px1, py1, px2, py2)            
            ps = s
            px1, py1, px2, py2 = line[0]
                            
    if belt == False:
        print("No Seatbelt detected")
        ans="No Seatbelt detected"

    # Show The "beltframe"
    cv2.imshow("Seat Belt", beltframe)
    return ans
def home():
    return render_template('seatbelt1.html')
if __name__ == '__main__':
    app.run(debug=True)

import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import pandas as pd
import pickle
from sklearn.datasets import load haarcascade_frontalface_alt
from sklearn.datasets import load haarcascade_lefteye_2splits
from sklearn.datasets import load haarcascade_righteye_2splits
from sklearn.model_selection import cnnCat2.h5
from xgboost import CascadeClassifier
from flask import Flask,render_template
X,y =load_model(return_X_y=True,as frame=True)
X.head()
X_train,Xtest,y_train,y_test=train_test_split(X,y,test_size)
app = Flask(__name__)
@app.route('/')
def test1():
    return render_template('drowsiness.html')
@app.route('/project')
def pro():
    ans=""



mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



lbl=['Close','Open']

model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
def home():
    return render_template('drowsiness1.html')
if __name__ == '__main__':
    app.run(debug=True)

