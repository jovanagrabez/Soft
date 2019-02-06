import cv2
import vector_prof
import numpy as np
import math
from neuronska import napravi_model, recognise, find, recNum
from detekcija import search_for_blue, search_for_green

f = open('out.txt', 'a')
f.write('RA 154/2015 Jovana Grabez\n')
f.write('file sum')
f.close()



for p in range(0,10):
    kernel = np.ones((2, 2))
    NameofVideo = 'video-'+str(p)
    video = cv2.VideoCapture(NameofVideo + '.avi')
    FrameLoad, frame = video.read()
    model = napravi_model()
    model.load_weights(''
    'weights.h5')

    
    LinePoints = search_for_blue(cv2.morphologyEx(frame,cv2.MORPH_OPEN,kernel=kernel))
    Zelene=search_for_green(cv2.morphologyEx(frame,cv2.MORPH_OPEN,kernel=kernel))
 

   
    sum1 = 0
    sum2=0
   
    AllN = []
    NumF = 0
    while (True):
        FrameLoad, frame = video.read()
        if FrameLoad is not True:
            break
        shape = recognise(frame)
        for i, kontura in enumerate(shape):
            (x, y, w, h) = kontura
            brooj = {'centar': (int(x + w / 2), int(y + h / 2)), 'brojFrejma': NumF}
            findIt = find(AllN, brooj)
            if len(findIt) != 0:
                index = findIt[0]
                ist = {'brojFrejma':NumF,'centar': brooj['centar']}
                AllN[index]['brojFrejma'] = NumF
                AllN[index]['centar'] = brooj['centar']
                
            else: 
                brooj['value'] = recNum(frame, kontura, model)
                brooj['crossLine'] = False
                brooj['presaoPlavu']=False
                brooj['presaoZelenu']=False
                AllN.append(brooj)
               
        for brooj in AllN:
            if (NumF - brooj['brojFrejma']) > 3:
                continue
            distanca, jjj, r = vector_prof.pnt2line(brooj['centar'], LinePoints[0], LinePoints[1])
            blabla,nes,ppp=vector_prof.pnt2line(brooj['centar'], Zelene[0], Zelene[1])
           
            if  not  brooj['presaoPlavu'] :         
                if distanca < 10.0 and r == 1:
                        sum1 += int(brooj['value'])
                        brooj['presaoPlavu'] =True      
                        
            if  not brooj['presaoZelenu']:
                if blabla < 10.0 and ppp == 1:
                    sum1 -= int(brooj['value']) 
                    brooj['presaoZelenu']=True
        
            cv2.putText(frame, "Suma: " + str(sum1), (12, 25), cv2.FONT_HERSHEY_COMPLEX,
                        0.5,(201, 95, 117), 1)
            cv2.putText(frame, "Suma2: " + str(sum2), (20, 35), cv2.FONT_HERSHEY_COMPLEX,
                        0.5,(201, 95, 117), 1)
            cv2.putText(frame, "Konacna: " + str(sum1-sum2), (45, 50), cv2.FONT_HERSHEY_COMPLEX,
                        0.5,(201, 95, 117), 1)
            cv2.circle(frame, brooj['centar'], 17, [201, 95, 117], 1)
           

        cv2.imshow(NameofVideo, frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        NumF += 1

    cv2.destroyAllWindows()
    video.release()

    f = open('out.txt', 'a')
    f.write('\n' + NameofVideo + '.avi' + '\t' + str(sum1))
    f.close()