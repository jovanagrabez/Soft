import cv2
import vector_prof
import numpy as nmp
import math

def search_for_blue(image):
    #funkcija za pronalazenje plave linije
    #postavljanje granica
    frame_for_blue = image.copy()

    frame_for_blue[:, :, 1] = 0

    gray = cv2.cvtColor(frame_for_blue, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

  #  down_gr = nmp.array([100, 50, 50])
   # up_gr = nmp.array([130, 255, 255])
    #maskica = cv2.inRange( cv2.cvtColor(image, cv2.COLOR_BGR2HSV), down_gr, up_gr)
   # edges = cv2.Canny(cv2.bitwise_and(image, image, mask=maskica), 50, 200, None, 3)
   #pronalazenje linije metodom houg
    blue_line = cv2.HoughLinesP(th, 1, nmp.pi / 180, 50, None, 50, 10)
    end_points= [(0,0),(0,0)]
    duzinaa = 0
    
    #ovaj dio racuna krajnje tacke linije
    if not (blue_line is None):
        for index,l in enumerate(blue_line):
            line = blue_line[index][0]
            x2=line[2]
            x1=line[0]
            y2=line[3]
            y1=line[1]
            length_of_line = math.sqrt((x2 -x1) ** 2
                                 + (y2 - y1) ** 2)
            if duzinaa< length_of_line:
                end_points[0] = (x1, y1)
                end_points[1] = (x2, y2)
                duzinaa = length_of_line

    return end_points


def search_for_green(image):
    frame_for_blue = image.copy()

    frame_for_blue[:, :, 0] = 0

    gray = cv2.cvtColor(frame_for_blue, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

  #  down_gr = nmp.array([100, 50, 50])
   # up_gr = nmp.array([130, 255, 255])
    #maskica = cv2.inRange( cv2.cvtColor(image, cv2.COLOR_BGR2HSV), down_gr, up_gr)
   # edges = cv2.Canny(cv2.bitwise_and(image, image, mask=maskica), 50, 200, None, 3)
   #pronalazenje linije metodom houg
    blue_line = cv2.HoughLinesP(th, 1, nmp.pi / 180, 50, None, 50, 10)
    end_points= [(0,0),(0,0)]
    duzinaa = 0
    
    #ovaj dio racuna krajnje tacke linije
    if not (blue_line is None):
        for index,l in enumerate(blue_line):
            line = blue_line[index][0]
            x2=line[2]
            x1=line[0]
            y2=line[3]
            y1=line[1]
            length_of_line = math.sqrt((x2 -x1) ** 2
                                 + (y2 - y1) ** 2)
            if duzinaa< length_of_line:
                end_points[0] = (x1, y1)
                end_points[1] = (x2, y2)
                duzinaa = length_of_line

    return end_points
