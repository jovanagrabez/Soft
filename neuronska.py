import numpy as np
import math
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.datasets import mnist
from model import napravi_model
import cv2





def find(allNum, element):
    found = []
    for ind,elm in enumerate(allNum):
        (eX, eY) = element['centar']
        (x, y) = elm['centar']
        distance = math.sqrt(math.pow((x - eX), 2) + math.pow((y - eY), 2))
        if distance< 20:
            found.append(ind)
    return found



def recognise(img):
   
    mask = cv2.inRange(img, np.array([230, 230, 230]), np.array([255, 255, 255]))
    img = cv2.bitwise_and(img, img, mask=mask)
    siviloo = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blury = cv2.GaussianBlur(siviloo, (5, 5), 0)
  
    imm, cont, nista = cv2.findContours(blury.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cont_num = []
    for conture in cont:
        (x, y, w, h) = cv2.boundingRect(conture)
        povrsina = cv2.contourArea(conture)
        if h > 12 :
            if povrsina >30:
                if povrsina<1000:
                    koordinate = (x, y, w, h)
                    cont_num.append(koordinate)
    return cont_num


def recNum(img, conture, classifier):
    siviloo = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (x, y, w, h) = conture
   
    
    numberr = siviloo[int(y + h / 2)-12:int(y + h / 2)+12, int(x + w / 2)-12:int(x + w / 2)+12]
    (tr, numberr) = cv2.threshold(numberr, 126, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    numberr = correctNum(numberr)
    numberr = cutt(numberr)
    br = classifier.predict_classes(numberr.reshape(1, 28, 28, 1))
    return int(br)






def cutt(numberr):
    areas = []
    makss = 0
    tacno, sivilo = cv2.threshold(numberr, 126, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(sivilo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    for con in contours:
        areas.append(cv2.contourArea(con))
        
   
 
    for ind,povrs in enumerate(areas):
        if areas[makss]<povrs:
            makss = ind
    [x, y, w, h] = cv2.boundingRect(contours[makss])
    vely=y + h + 1
    velx=x + w + 1
    cuted = numberr[y:vely, x:velx]
    cuted = cv2.resize(cuted, (28,28), interpolation=cv2.INTER_AREA)
    
    if len(areas) is None:
        return numberr
    return cuted



def correctNum(img):
    mom = cv2.moments(img)
    if abs(mom['mu02']) <= 1e-2:
        return img.copy()
    MOMENT = np.float32([[1, mom['mu11'] / mom['mu02'], -0.5 * 28 *mom['mu11'] / mom['mu02']], [0, 1, 0]])
    img = cv2.warpAffine(img, MOMENT, (28, 28), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img



#https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
  

        
    x_train = x_train/ 255
    y_train =y_train/ 255

    model = napravi_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(x_train, to_categorical(y_train),epochs=20)
    model.evaluate(y_train, to_categorical(y_test), verbose=0)
    model.save_weights('weights.h5')
   


    
