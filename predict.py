import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2
model_path = "./model.h5"
model = load_model(model_path)
output_path = ""
img_path = ""
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
pic = cv2.imread(img_path)
gray = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray,1.3,5)
age_ = []
gender_ = []
for (x,y,w,h) in faces:
  img = gray[y-50:y+40+h,x-10:x+10+w]
  img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
  img = cv2.resize(img,(200,200))
  predict = model.predict(np.array(img).reshape(-1,200,200,3))
  age_.append(predict[0])
  gender_.append(np.argmax(predict[1]))
  gend = np.argmax(predict[1])
  if gend == 0:
    gend = 'Man'
    col = (255,0,0)
  else:
    gend = 'Woman'
    col = (203,12,255)
  cv2.rectangle(pic,(x,y),(x+w,y+h),(0,225,0),4)
  cv2.putText(pic,"Age : "+str(int(predict[0]))+" / "+str(gend),(x,y),cv2.FONT_HERSHEY_SIMPLEX,w*0.005,col,4)
pic1 = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
plt.imshow(pic1)
plt.show() 
print(age_,gender_)
cv2.imwrite(output_path,pic)
