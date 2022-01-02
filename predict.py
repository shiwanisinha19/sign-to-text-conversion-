import cv2
import numpy as np
from keras.models import model_from_json
import operator
import sys, os
import random
import string



from keras.models import load_model

from keras.models import model_from_json
with open("general1-bw.json","r") as file:
  general1_json=file.read()
  loaded_model=model_from_json(general1_json)
  loaded_model.load_weights("general1-bw.h5")
  print("Loaded model")

cam = cv2.VideoCapture(0)
# Category dictionary
categories = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

while(cam.isOpened()):
    
    
    ret, frame = cam.read()
    frame=cv2.flip(frame,1)
    
    
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,2)
    
    roi = frame[y1:y2, x1:x2]
     
 
    #cv2.imshow("Frame", frame)
    minvalue=20
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    #blur=cv2.GaussianBlur(roi,(5,5),2)

    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)

    #th3=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    #ret,test_image=cv2.threshold(th3,minvalue,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    roi=cv2.resize(roi,(128,128))
    cv2.imshow("sign detection",roi)

    result= loaded_model.predict(roi.reshape(1,128,128,1))
    prediction = {'A': result[0][0], 
                  'B': result[0][1], 
                  'C': result[0][2],
                  'D': result[0][3],
                  'E': result[0][4],
                  'F': result[0][5],
                  'G': result[0][6], 
                  'H': result[0][7], 
                  'I': result[0][8],
                  'J': result[0][9],
                  'K': result[0][10],
                  'L': result[0][11],
                  'M': result[0][12], 
                  'N': result[0][13], 
                  'O': result[0][14],
                  'P': result[0][15],
                  'Q': result[0][16],
                  'R': result[0][17],
                  'S': result[0][18], 
                  'T': result[0][19], 
                  'U': result[0][20],
                  'V': result[0][21],
                  'W': result[0][22],
                  'X': result[0][23],
                  'Y': result[0][24], 
                  'Z': result[0][25],}
                  
                  
                  
                  
                  
                  

      
    
      
    
    prediction=sorted(prediction.items(),key=operator.itemgetter(1),reverse=True)

    #(text_width, text_height) = cv2.getTextSize('Predicted text:      ', 1, fontScale=1.5, thickness=2)[0]
    #response=prediction

    #cv2.putText(frame, response , (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)
    
    frame = cv2.putText(frame, 'Predicted text: ', (50,70), cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 5, cv2.LINE_AA)
    #frame = cv2.putText(frame, prediction, (300,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
    #cv2.putText(frame, prediction,
#(170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    #cv2.putText(frame,result,cv2.FONT_HERSHEY_PLAIN,1,(0,128,128),1)
    #display_string=str(prediction)

    #cv2.putText(frame,prediction,cv2.FONT_HERSHEY_PLAIN,1,(0,128,128),1)
    cv2.putText(frame, prediction[0][0], (70, 200), cv2.FONT_HERSHEY_PLAIN, 10, (0,0,255), 10)    
    #cv2.putText(frame,f"sign says:{prediction}",cv2.FONT_HERSHEY_PLAIN,1,(0,128,128),1)
    cv2.imshow("FRAME",frame)

    interrupt=cv2.waitKey(10)
    if interrupt & 0xFF==27:
       break
        
cam.release()
cv2.destroyAllWindows()