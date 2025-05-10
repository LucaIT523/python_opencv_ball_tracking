import cv2
import numpy as np
from math import *
import time
import sys

#python ball_Genius.py demo1

WIDTH, HIGHT = (640, 360)
FRAME = 30.0
TIMEVIEW = 10
AREA = 65.0
SIZEMIN = 5.0
SIZEMAX = 20.0

g_StartX, g_StartY, g_EndX, g_EndY = (0,0,640,320)

INFILE = "demo4"#sys.argv[1]
OUTFILE = 'out_' + INFILE
cap = cv2.VideoCapture(INFILE + '.mp4')
FRAME = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
writer = cv2.VideoWriter(OUTFILE + '.avi', fourcc, 30.0, (WIDTH*2, HIGHT*2))

frameCount = 0

#Green Color
#green_light=np.array([50,60,0],np.uint8)
#green_dark=np.array([90,255,255],np.uint8)

#green_light=np.array([41,90,0],np.uint8)
#green_dark=np.array([149,255,255],np.uint8)

black_light=np.array([0,0,0],np.uint8)
black_dark=np.array([255,90,80],np.uint8)

#Red1 Color
red1_light=np.array([0,140,90],np.uint8)
red1_dark=np.array([5,255,255],np.uint8)

#Red2 Color
red2_light=np.array([150,120,100],np.uint8)
red2_dark=np.array([255,255,250],np.uint8)

#Yellow Color
yellow_light=np.array([20,150,100],np.uint8)
yellow_dark=np.array([40,255,255],np.uint8)

#Yellow2 Color
yellow2_light=np.array([10,180,180],np.uint8)
yellow2_dark=np.array([19,255,255],np.uint8)

#White Color
#white_light=np.array([10,0,200],np.uint8)
#white_dark=np.array([255,120,255],np.uint8)
#white_light=np.array([15,0,140],np.uint8)
#white_dark=np.array([255,100,255],np.uint8)
white_light=np.array([20,0,200],np.uint8)
white_dark=np.array([255,110,255],np.uint8)

# ball tracking class
class CBall:
    def __init__(self, b, g, r, name):
        self.ballColor = {'r': 0, 'g': 0, 'b': 0}
        self.ballPosition = {'x': 0, 'y': 0, 'r': 0}
        self.ballTrack = []          
        self.ballColor['r'], self.ballColor['g'], self.ballColor['b'] = r, g, b
        self.ballName = name   
        self.distanceThreshold = 5
        self.countStop = 0    
        self.stopTrack =10
        self.img = []
        self.imgPos = []
        self.imgArea = 0
    
    def getColor(self):
        return (self.ballColor['b'], self.ballColor['g'], self.ballColor['r'])
    def getPos(self):
        return (self.ballPosition['x'], self.ballPosition['y'], self.ballPosition['r'])
    def getTracks(self):
        return self.ballTrack
    def getName(self):
        return self.ballName
    def getImg(self):
        return self.img
    def getGrayImg(self, mask):
        th2 = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        th2 = cv2.adaptiveThreshold(th2,255,cv2.ADAPTIVE_THRESH_MEAN_C,mask,11,2)
        return th2
    def getImgPos(self) :
        return self.imgPos
    def setImg(self, img):
        r = self.ballPosition['r'] + 3
        x1 = self.ballPosition['x'] - r
        y1 = self.ballPosition['y'] - r
        x2 = self.ballPosition['x'] + r
        y2 = self.ballPosition['y'] + r 
        self.imgPos = (x1, y1, x2, y2)
        self.img = img[y1:y2, x1:x2]
    def getImgArea(self):
        return self.imgArea
    def updatePos(self, x, y, r, area):
        self.imgArea = area
        self.ballPosition['x'], self.ballPosition['y'], self.ballPosition['r'] = x, y, r 
        self.ballTrack[len(self.ballTrack)-1]['x'] = self.ballPosition['x']
        self.ballTrack[len(self.ballTrack)-1]['y'] = self.ballPosition['y']
        self.ballTrack[len(self.ballTrack)-1]['r'] = self.ballPosition['r']
        
    def setPos(self, x, y, r, area):
        self.ballPosition['r'] = r
        if area == 0 :
            # update track
            self.ballTrack.append({'x': self.ballPosition['x'], 'y': self.ballPosition['y'], 'r': self.ballPosition['r']})
            return
        # calculate distance of moving
        self.imgArea = area
        distance = sqrt(pow(x-self.ballPosition['x'], 2) + pow(y-self.ballPosition['y'], 2))
        if distance > self.distanceThreshold:
            # update position
            self.ballPosition['x'], self.ballPosition['y'] = x, y   
            self.countStop = 0
        elif self.countStop < self.stopTrack :
            #stop state
            w_distance = 0
            if len(self.ballTrack) > 0 and self.ballTrack[len(self.ballTrack)-1-self.countStop]['x'] > 0 :
                w_distance = sqrt(pow(x-self.ballTrack[len(self.ballTrack)-1-self.countStop]['x'], 2) + pow(y-self.ballTrack[len(self.ballTrack)-1-self.countStop]['y'], 2))
            if w_distance < self.distanceThreshold : 
                self.countStop += 1
            else :
                self.countStop = 0
        # update track
        self.ballTrack.append({'x': self.ballPosition['x'], 'y': self.ballPosition['y'], 'r': self.ballPosition['r']})
        if self.countStop >= self.stopTrack :
            return 1
        else :
            return 0
    def resetTrack(self):
        if len(self.ballTrack) > 0 :
            self.ballTrack.clear()
        
def calc_Pane_Info_func(img):
    wWIDTH, whight = (np.int8(WIDTH/10), np.int8(HIGHT/10))
    backImg = cv2.resize(img, (wWIDTH, whight), interpolation = cv2.INTER_LINEAR)
    b, g, r=cv2.split(backImg)
    backImg = cv2.cvtColor(backImg, cv2.COLOR_BGR2GRAY)

    for j in range(wWIDTH):
        for i in range(whight):
            w_R = np.int16(r[i,j])
            w_G = np.int16(g[i,j])
            w_B = np.int16(b[i,j])
            if w_G+w_G > w_R+w_B:
                green = w_G + w_G - w_R - w_B
                if(green >= 255):
                    green = 255                    
            else :
                green = 0                               
            backImg[i, j] = green
    ret,backImg = cv2.threshold(backImg,100,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(backImg, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)  
    
    #cv2.imwrite("backImg.jpg", backImg)
    

    for i in range(len(contours)):   
        cnt = contours[i]   
        if(len(cnt)<15):
            continue
        w_X,w_Y,w_W,w_H = cv2.boundingRect(cnt)

        w_X = w_X*10+3
        w_Y = w_Y*10+3
        w_W = w_W*10-3
        w_H = w_H*10-3                
        break       
    return (w_X, w_Y, w_W+w_X, w_H+w_Y)

def calc_Ball_Region_func(img, inImg, ball):
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(grayImg,27,255,cv2.THRESH_BINARY)
    grayImg = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    contours,hierarchy = cv2.findContours(thresh[g_StartY:g_EndY, g_StartX:g_EndX], cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
    w_Regions= []
    w_Region = [0,0,0,0]
    for i in range(len(contours)):   
        cnt = contours[i]   
        area = cv2.contourArea(cnt)
        if area < AREA:
            continue              
        (x,y),radius = cv2.minEnclosingCircle(cnt)   

        if radius < SIZEMIN or radius > SIZEMAX:
            continue     
        #print("contours:",ball.getName(),"-",i,":",int(x+0.5)+g_StartX,int(y+0.5)+g_StartY, int(radius+0.5), area)
        w_Region = [int(x+0.5)+g_StartX,int(y+0.5)+g_StartY, int(radius+0.5), area]
        w_Regions.append(w_Region)
    if len(w_Regions) > 1 :
        x, y, r = ball.getPos()
        k = 0
        temp_sum = np.int32(0)
        temp_max = np.int32(9999)
        for i in range(len(w_Regions)) :
            temp_sum = abs(w_Regions[i][0] - x) + abs(w_Regions[i][1] - y) + abs(w_Regions[i][2] - r) 
            if(temp_sum < temp_max) :
                temp_max = temp_sum
                k = i
        w_Region = w_Regions[k]
        #print("contour:",ball.getName(),"-",i,":",w_Region[0],w_Region[1], w_Region[2], w_Region[3])
    if(frameCount > 1  and w_Region[3] > 0) :
        x,y,r=ball.getPos()
        if(r>0):
            w_Area0 = ball.getImgArea()
            x1,y1,x2,y2 = (ball.getImgPos())
            r = int((x2-x1)/2)-3
            #old position Area
            x, y, r, w_Area1 = calc_Area_Ball_func((x1,y1,x2,y2), thresh[y1:y2,x1:x2])
            x +=x1
            y +=y1
            #current Area
            w_Area2 = w_Region[3]
            #print("old:",ball.getName(),"-",ball.getPos(), w_Area0)
            #print("oldpos:",ball.getName(),"-",int(x+0.5), int(y+0.5), int(r+0.5), w_Area1)
            #print("curps:",ball.getName(),"-",w_Region[0],w_Region[1], w_Region[2], w_Region[3])
            if w_Area1 > 0 and abs(w_Area0 - w_Area1) < abs(w_Area0 - w_Area2) :            
                return (int(x+0.5), int(y+0.5), int(r+0.5)), w_Area1, grayImg
            else :
                return (w_Region[0],w_Region[1],w_Region[2]), w_Area2, grayImg
    return (w_Region[0],w_Region[1],w_Region[2]), w_Region[3], grayImg

def calc_Area_Ball_func(region, grayImg):
    contours,hierarchy = cv2.findContours(grayImg, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
    x = 0
    y = 0
    r = 0
    w_Area = 0
    maxArea = 0
    for i in range(len(contours)):   
        cnt = contours[i] 
        w_Area = cv2.contourArea(cnt)
        (x1,y1),r1 = cv2.minEnclosingCircle(cnt)       
        if r1 < SIZEMIN or r1 > SIZEMAX +3:
            continue         
        if maxArea < w_Area :
            maxArea = w_Area
            x, y, r = x1, y1, r1
    return x, y, r, maxArea
        
def resetTracks():
    redBall.resetTrack()
    yellowBall.resetTrack()
    whiteBall.resetTrack()    
    


redBall = CBall(0, 0, 255, 'Red')
yellowBall = CBall(0, 255, 255, 'Yellow')
whiteBall = CBall(255, 255, 255, 'White')

#Init : calculate info of BallPane 
startTime = time.time()
ret, frame = cap.read()
if ret:
    inputImg = cv2.resize(frame, (WIDTH, HIGHT), interpolation = cv2.INTER_LINEAR)
    g_StartX, g_StartY, g_EndX, g_EndY = calc_Pane_Info_func(inputImg)
    w_RateX = (g_EndX - g_StartX)/WIDTH
    w_Rate = (g_EndX - g_StartX)*(g_EndY - g_StartY)/(WIDTH*HIGHT)
    AREA = AREA * w_Rate
    SIZEMIN = SIZEMIN * w_RateX
    SIZEMAX = SIZEMAX * w_RateX
    print("Info of Pane(sX,sY,eX,eY) : ", g_StartX, g_StartY, g_EndX, g_EndY)  
endTime = time.time()
print('time', frameCount, ':', endTime - startTime)  

stopState = 0
stopTime = 0
redBallStop = 0
yellowBallStop = 0
whiteBallStop = 0

       
while (True):  

    if ret:
        frameCount +=1        
        startTime = time.time()

        inputImg = cv2.resize(frame, (WIDTH, HIGHT), interpolation = cv2.INTER_LINEAR)    
        HSVImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2HSV)
        
        BrightImg = HSVImg[g_StartY:g_EndY, g_StartX:g_EndX]
        
        #w_H = BrightImg[...,0].mean()
        w_S = BrightImg[...,1].mean()
        w_V = BrightImg[...,2].mean()
        
        w_S1 = 0
        w_S2 = 255
        #print("means",w_S, w_V)
        if w_S < 50 :
            w_S1 = 50
            w_S2 = w_S + 50
        elif w_S > 205 :
            w_S1 = w_S - 50
            w_S2 = 255
        
        green_light=np.array([41,w_S1,0],np.uint8)
        green_dark=np.array([149,w_S2,255],np.uint8)
        
        #BallPanel : Range Specification
        green=cv2.inRange(HSVImg,green_light,green_dark)
        #balck=cv2.inRange(HSVImg,black_light,black_dark)
        # combine the mask        
        #green = cv2.bitwise_or(green, balck)  
        
        red1=cv2.inRange(HSVImg,red1_light,red1_dark) 
        red2=cv2.inRange(HSVImg,red2_light,red2_dark)
        yellow=cv2.inRange(HSVImg,yellow_light,yellow_dark)
        yellow2=cv2.inRange(HSVImg,yellow2_light,yellow2_dark)
        white=cv2.inRange(HSVImg,white_light,white_dark)
        black=cv2.inRange(HSVImg,black_light,black_dark)
        # combine the mask        
        other = cv2.bitwise_or(red1, red2)   
        other = cv2.bitwise_or(other, yellow)   
        other = cv2.bitwise_or(other, yellow2) 
        other = cv2.bitwise_or(other, white) 
        #other = cv2.bitwise_or(other, black) 
        
        res_green = cv2.bitwise_and(inputImg,inputImg, mask=other)
        
        green_gray = cv2.cvtColor(res_green, cv2.COLOR_BGR2GRAY)
        ret,green_gray = cv2.threshold(green_gray,30,255,cv2.THRESH_BINARY)
        
        contours,hierarchy = cv2.findContours(green_gray[g_StartY:g_EndY, g_StartX:g_EndX], cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE) #0 , 1)
        w_hight,  w_width= g_EndY-g_StartY, g_EndX-g_StartX
        w_Regions= []
        w_Region = [0,0,0,0]        
        for i in range(len(contours)):   
            cnt = contours[i]   
            area = cv2.contourArea(cnt)
            if area < AREA:
                continue              
            (x,y),radius = cv2.minEnclosingCircle(cnt)  
            
            if x-radius < 2 or x+radius > w_width-2 or y-radius < 2 or y+radius > w_hight-2 or radius < SIZEMIN or radius > SIZEMAX+8 :
                continue     
           
            w_Region = [int(x+0.5) +g_StartX ,int(y+0.5) + g_StartY, int(radius+0.5), area]
            w_Regions.append(w_Region)   
        green_gray = cv2.cvtColor(green_gray, cv2.COLOR_GRAY2BGR)
        
        # create a black image
        result = np.zeros((HIGHT, WIDTH, 3), dtype = np.uint8)
        #result = green_gray.copy()
        
        for i in range(len(w_Regions)) :
            #cv2.circle(result, (w_Regions[i][0], w_Regions[i][1]), w_Regions[i][2]+3, (0, 0, 255), 1);
            r = 18 # w_Regions[i][2] +3
            x,y,w,h=w_Regions[i][0]-r,w_Regions[i][1]-r,w_Regions[i][0]+r,w_Regions[i][1]+r
            cv2.imwrite("longshot/"+INFILE+"_"+str(frameCount)+"_"+str(i)+".jpg", inputImg[y:h, x:w])
            result[y:h, x:w]=inputImg[y:h, x:w]
            
      
        
        
        
    #RedBall : Range Specification
        red1=cv2.inRange(HSVImg,red1_light,red1_dark) 
        red2=cv2.inRange(HSVImg,red2_light,red2_dark)
        # combine the mask        
        red = cv2.bitwise_or(red1, red2)        
        res_red = cv2.bitwise_and(inputImg,inputImg, mask=red)
        
        region, Area, res_red = calc_Ball_Region_func(res_red, inputImg, redBall)
        if Area > 0 :
            x, y, r = (region)
            redBallStop = redBall.setPos(x, y, r, Area)
            redBall.setImg(inputImg)
            if stopState == 1 and redBallStop==0:
                resetTracks()  
                stopState = 0              
        else :
            redBallStop = redBall.setPos(0,0,0,0)
        if w_V > 70 :            
            #YellowBall : Range Specification
            yellow=cv2.inRange(HSVImg,yellow_light,yellow_dark) 
            res_yellow = cv2.bitwise_and(inputImg,inputImg, mask=yellow) 
            region, Area, res_yellow = calc_Ball_Region_func(res_yellow, inputImg, yellowBall)
            if Area > 0 :
                x, y, r = (region)
                yellowBallStop = yellowBall.setPos(x, y, r, Area)
                yellowBall.setImg(inputImg)
                if stopState == 1 and yellowBallStop==0:
                    resetTracks()  
                    stopState = 0                 
            else :
                yellowBallStop = yellowBall.setPos(0,0,0,0)        
          
            #WhiteBall : Range Specification
            white=cv2.inRange(HSVImg,white_light,white_dark) 
            res_white = cv2.bitwise_and(inputImg,inputImg, mask=white) 
            region, Area, res_white = calc_Ball_Region_func(res_white, inputImg, whiteBall)
        
            if Area > 0 :
                x, y, r = (region)
                whiteBallStop = whiteBall.setPos(x, y, r, Area) 
                whiteBall.setImg(inputImg)
                if stopState == 1 and whiteBallStop==0:
                    resetTracks()  
                    stopState = 0                 
            else :
                whiteBallStop = whiteBall.setPos(0,0,0,0)   
        else :
            #YellowBall : Range Specification
            yellow=cv2.inRange(HSVImg,yellow2_light,yellow2_dark) 
            res_yellow = cv2.bitwise_and(inputImg,inputImg, mask=yellow) 
            region, Area, res_yellow = calc_Ball_Region_func(res_yellow, inputImg, yellowBall)
            if Area > 0 :                
                x, y, r = (region)
                yellowBallStop = yellowBall.setPos(x, y, r, Area)
                yellowBall.setImg(inputImg)
                if stopState == 1 and yellowBallStop==0:
                    resetTracks()  
                    stopState = 0                
            else :
                yellowBallStop = yellowBall.setPos(0,0,0,0)        
            
            #WhiteBall : Range Specification
            white=cv2.inRange(HSVImg,yellow_light,yellow_dark) 
            res_white = cv2.bitwise_and(inputImg,inputImg, mask=white) 
            region, Area, res_white = calc_Ball_Region_func(res_white, inputImg, whiteBall)
        
            if Area > 0 :
                x, y, r = (region)
                whiteBallStop = whiteBall.setPos(x, y, r, Area) 
                whiteBall.setImg(inputImg)
                if stopState == 1 and whiteBallStop==0:
                    resetTracks()  
                    stopState = 0                
            else :
                whiteBallStop = whiteBall.setPos(0,0,0,0)   
            
            
        if stopState == 0 and redBallStop == 1 and yellowBallStop == 1 and whiteBallStop == 1 :
            stopState = 1
            #stopTime = time.time()
        print("stopLog:", stopState, redBallStop,yellowBallStop, whiteBallStop);
        #if stopState == 1 and time.time() - stopTime > TIMEVIEW :
            #resetTracks()
        font = cv2.FONT_HERSHEY_SIMPLEX        
        x, y, r = redBall.getPos()
        if r > 0 :
            print("redPosLog",x, y, r);
            r += 5
            cv2.line(inputImg, (x-r,y), (x+r,y), (255, 0, 0), 2)
            cv2.line(inputImg, (x,y-r), (x,y+r), (255, 0, 0), 2)
            cv2.putText(inputImg, redBall.getName() ,(x+r,y+r), font, 1,(redBall.getColor()),2,cv2.LINE_AA)            
        x, y, r = yellowBall.getPos()
        if r > 0 :
            r += 5
            cv2.line(inputImg, (x-r,y), (x+r,y), (255, 0, 0), 2)
            cv2.line(inputImg, (x,y-r), (x,y+r), (255, 0, 0), 2)
            cv2.putText(inputImg, yellowBall.getName() ,(x+r,y+r), font, 1,(yellowBall.getColor()),2,cv2.LINE_AA)
        x, y, r = whiteBall.getPos()
        if r > 0 :
            r += 5
            cv2.line(inputImg, (x-r,y), (x+r,y), (255, 0, 0), 2)
            cv2.line(inputImg, (x,y-r), (x,y+r), (255, 0, 0), 2) 
            cv2.putText(inputImg, whiteBall.getName() ,(x+r,y+r), font, 1,(whiteBall.getColor()),2,cv2.LINE_AA)
        
        cv2.putText(inputImg, str(frameCount) ,(10, 30), font, 1,(whiteBall.getColor()),2,cv2.LINE_AA)
        redTrack = redBall.getTracks()
        yellowTrack = yellowBall.getTracks()
        whiteTrack = whiteBall.getTracks()
        if len(redTrack) > 1 :
            for i in range(len(redTrack)-1):
                if(redTrack[i]['x'] > 0):
                    cv2.line(inputImg, (redTrack[i]['x'],redTrack[i]['y']), (redTrack[i+1]['x'],redTrack[i+1]['y']), (redBall.getColor()),1)
                if(yellowTrack[i]['x'] > 0):
                    cv2.line(inputImg, (yellowTrack[i]['x'],yellowTrack[i]['y']), (yellowTrack[i+1]['x'],yellowTrack[i+1]['y']), (yellowBall.getColor()),1)
                if(whiteTrack[i]['x'] > 0):
                    cv2.line(inputImg, (whiteTrack[i]['x'],whiteTrack[i]['y']), (whiteTrack[i+1]['x'],whiteTrack[i+1]['y']), (whiteBall.getColor()),1)

        #writer.write(inputImg)
        
        imccn1 = np.concatenate((inputImg, res_green), axis=1)
        imccn2 = np.concatenate((result, green_gray), axis=1)
        imccn3 = np.concatenate((imccn1, imccn2), axis=0)
        writer.write(imccn3)

        endTime = time.time()
        print('time', frameCount, ':', endTime - startTime)   
        
        #if frameCount > 2160:
            #break

    else:
        break
    ret, frame = cap.read()
cap.release()
writer.release()
print('End')