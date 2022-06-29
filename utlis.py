import cv2
import numpy as np
from pyparsing import And, White
 
def getContours(img,cThr=[100,100],showCanny=False,minArea=10000,filter=0,draw =False):
    total = 1000000
    Border=cv2.copyMakeBorder(img,2,2,2,2,cv2.BORDER_CONSTANT,value=[255,255,255])
    imgGray = cv2.cvtColor(Border,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,cThr[0],cThr[1])
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=3)
    imgThre = cv2.erode(imgDial,kernel,iterations=2)
    if showCanny:cv2.imshow('Canny',imgThre)
    contours,hiearchy = cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i,True) 
            print(peri/100,"perimeter")
            p= peri/800
            print(p, "radius")
            volume= (4/3*3.14*p*p*p,"cm3")
            print(volume)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append([len(approx),area,approx,bbox,i])
            else:
                finalCountours.append([len(approx),area,approx,bbox,i])
    finalCountours = sorted(finalCountours,key = lambda x:x[1] ,reverse= True)
    if draw:
        for con in finalCountours:
            cv2.drawContours(Border,con[4],-1,(0,0,255),3)
    return img, finalCountours
 
def reorder(myPoints):
    #print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew
 
def warpImg (img,points,w,h,pad=5):
    # print(points)
    points =reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad,pad:imgWarp.shape[1]-pad]
    return imgWarp
 
def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5

