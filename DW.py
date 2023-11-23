# BESM ALLAH ALRAHMAN ALRAHEM
# Modified by:
import time
from djitellopy import tello
import cv2
import numpy as np
from threading import Thread
#Imported libraries

def empty(a):
    pass
####################################################################################
### MY VARIABLES ###################################################################
width = 320         #640 # width of the image
height =240         #480 # height of the image      #50 #100
deadZoneDW = 5     #50 #100
####################################################################################
frameWidth = width
frameHeight = height
global imgContourFW
global myDirection ; global myDirection_LR
global imgContourDW
global myDirectionDW ; global myDirectionDW_LR
global Landing_search ; global Search_right_side; global Search_left_side
myDirection = 0
myDirection_LR = 0
myDirectionDW = 0
myDirectionDW_LR = 0
Landing_search = 0 ; Search_right_side = 0; Search_left_side = 0

global keepShowingDW;keepShowingDW = True
global left_right_velocity; global for_back_velocity; global up_down_velocity; global yaw_velocity
left_right_velocity = 0; for_back_velocity = 0; up_down_velocity = 0; yaw_velocity = 0
####################################################################################

threshold1D = 150
threshold2D = 199
####################################################################################
### MY FUNCTIONS ###################################################################

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y],(0, 0), None, scale, scale)
                else:###### correction: imgArray[x]--> imgArray[x][y]
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8) ###### correction:imgeBlank --> imageBlank
        hor = [imageBlank]*rows
        hor_con =  [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

def displayDW(img):# the cyan cells
    cv2.line(img,(int(frameWidth/2)-deadZoneDW,0),(int(frameWidth/2)-deadZoneDW,frameHeight),(255,255,0),3)# 1st cyan vertical line
    cv2.line(img,(int(frameWidth/2)+deadZoneDW,0),(int(frameWidth/2)+deadZoneDW,frameHeight),(255,255,0),3)# 2nd cyan vertical line
    cv2.circle(img,(int(frameWidth/2),int(frameHeight/2)),5,(0,0,255),5) # red circle
    cv2.line(img, (0,int(frameHeight / 2)-deadZoneDW), (frameWidth,int(frameHeight / 2) - deadZoneDW), (255, 255, 0), 3) # 1st cyan horizontal line
    cv2.line(img, (0, int(frameHeight / 2) + deadZoneDW), (frameWidth, int(frameHeight / 2) + deadZoneDW), (255, 255, 0), 3) # 2nd cyan horizontal line
#the cyan cells
def getContoursDW (img) :#,imgContourDW):
    global imgContourDW
    global myDirectionDW
    global myDirectionDW_LR
    global Landing_search
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = 3000
        if area > areaMin:
            cv2.drawContours(imgContourDW, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
            print(len(approx))
            objCor= len(approx)
            if objCor>9: # near circle
                x , y , w , h = cv2.boundingRect(approx)
                cx = int(x + (w / 2))   # the x of the center of the object
                cy = int(y + (h / 2))   # the y of the center of the object
                myfont= cv2.FONT_HERSHEY_SIMPLEX
                Landing_search =1#####################################################
                if (cy < int(frameHeight / 2) - deadZoneDW) and (cx > int(frameWidth / 2) + deadZoneDW):
                    cv2.putText(imgContourDW, " GO FORWARD AND GO RIGHT", (20, 50),myfont,1,(0, 0, 255), 3)
                    cv2.rectangle(imgContourDW,(int(frameWidth/2-deadZoneDW),0),(int(frameWidth/2+deadZoneDW),int(frameHeight/2)-deadZoneDW),(0,0,255),cv2.FILLED)
                    cv2.rectangle(imgContourDW,(int(frameWidth/2+deadZoneDW),int(frameHeight/2-deadZoneDW)),(frameWidth,int(frameHeight/2)+deadZoneDW),(0,0,255),cv2.FILLED)
                    myDirectionDW = 1 ; myDirectionDW_LR =1###########################
                elif (cy < int(frameHeight / 2) - deadZoneDW) and (cx <int(frameWidth/2)-deadZoneDW): #GO FORWARD and GO LEFT
                    cv2.putText(imgContourDW, " GO FORWARD AND GO LEFT", (20, 50),myfont,1,(0, 0, 255), 3)
                    cv2.rectangle(imgContourDW,(int(frameWidth/2-deadZoneDW),0),(int(frameWidth/2+deadZoneDW),int(frameHeight/2)-deadZoneDW),(0,0,255),cv2.FILLED)
                    cv2.rectangle(imgContourDW,(0,int(frameHeight/2-deadZoneDW)),(int(frameWidth/2)-deadZoneDW,int(frameHeight/2)+deadZoneDW),(0,0,255),cv2.FILLED)
                    myDirectionDW = 2 ; myDirectionDW_LR = 2#########################
                elif (cy > int(frameHeight / 2) + deadZoneDW) and (cx > int(frameWidth / 2) + deadZoneDW):
                    cv2.putText(imgContourDW, " GO BACKWARD AND RIGHT",(20, 50), myfont, 1,(0, 0, 255), 3)
                    cv2.rectangle(imgContourDW,(int(frameWidth/2-deadZoneDW),int(frameHeight/2)+deadZoneDW),(int(frameWidth/2+deadZoneDW),frameHeight),(0,0,255),cv2.FILLED)
                    cv2.rectangle(imgContourDW,(int(frameWidth/2+deadZoneDW),int(frameHeight/2-deadZoneDW)),(frameWidth,int(frameHeight/2)+deadZoneDW),(0,0,255),cv2.FILLED)
                    myDirectionDW = 3 ; myDirectionDW_LR = 3###################################
                elif (cy > int(frameHeight / 2) + deadZoneDW) and (cx <int(frameWidth/2)-deadZoneDW):
                    cv2.putText(imgContourDW, " GO BACKWARD ِAND LEFT",(20, 50), myfont, 1,(0, 0, 255), 3)
                    cv2.rectangle(imgContourDW,(int(frameWidth/2-deadZoneDW),int(frameHeight/2)+deadZoneDW),(int(frameWidth/2+deadZoneDW),frameHeight),(0,0,255),cv2.FILLED)
                    cv2.rectangle(imgContourDW,(0,int(frameHeight/2-deadZoneDW)),(int(frameWidth/2)-deadZoneDW,int(frameHeight/2)+deadZoneDW),(0,0,255),cv2.FILLED)
                    myDirectionDW = 4 ; myDirectionDW_LR = 4
                elif (cx <int(frameWidth/2)-deadZoneDW):
                    cv2.putText(imgContourDW, " GO LEFT " , (20, 50), myfont,1,(0, 0, 255), 3)
                    cv2.rectangle(imgContourDW,(0,int(frameHeight/2-deadZoneDW)),(int(frameWidth/2)-deadZoneDW,int(frameHeight/2)+deadZoneDW),(0,0,255),cv2.FILLED)
                    myDirectionDW = 1####################################################
                elif (cx > int(frameWidth / 2) + deadZoneDW):
                    cv2.putText(imgContourDW, " GO RIGHT ", (20, 50), myfont,1,(0, 0, 255), 3)
                    cv2.rectangle(imgContourDW,(int(frameWidth/2+deadZoneDW),int(frameHeight/2-deadZoneDW)),(frameWidth,int(frameHeight/2)+deadZoneDW),(0,0,255),cv2.FILLED)
                    myDirectionDW = 2
                elif (cy < int(frameHeight / 2) - deadZoneDW):
                    cv2.putText(imgContourDW, " GO FORWARD", (20, 50),myfont,1,(0, 0, 255), 3)
                    cv2.rectangle(imgContourDW,(int(frameWidth/2-deadZoneDW),0),(int(frameWidth/2+deadZoneDW),int(frameHeight/2)-deadZoneDW),(0,0,255),cv2.FILLED)
                    myDirectionDW = 3
                elif (cy > int(frameHeight / 2) + deadZoneDW):
                    cv2.putText(imgContourDW, " GO BACKWARD",(20, 50), myfont, 1,(0, 0, 255), 3)
                    cv2.rectangle(imgContourDW,(int(frameWidth/2-deadZoneDW),int(frameHeight/2)+deadZoneDW),(int(frameWidth/2+deadZoneDW),frameHeight),(0,0,255),cv2.FILLED)
                    myDirectionDW = 4
                else:
                    myDirectionDW=0 ; myDirectionDW_LR = 0

                cv2.rectangle(imgContourDW, (x, y), (x + w, y + h), (0, 255, 0), 5)
                cv2.putText(imgContourDW, "Center: ", (x+10, y+20),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(imgContourDW, "( " + str(int(cx)) + ", " + str(int(cy)) + " )", (cx, y+10),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
#This function set up coordinates of the object  이 함수는 물체의 좌표를 설정합니다

####################################################################################
### MY THREADS FOR REAL TIME IMAGE PROCESSING ######################################
def videoLiveShowDW():
    global keepShowingDW
    global imgContourDW
    global myDirectionDW
    global left_right_velocity;global for_back_velocity;global up_down_velocity;global yaw_velocity
    global Landing_search;global Search_right_side;global Search_left_side

    while keepShowingDW:
        # Display the resulting frame
        img = me.get_frame_read().frame[0:240, :]
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.resize(img, (320, 240))
        imgContourDW = img.copy()
        ## Object Detection
        # 1- Get grayimage
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2- Add Blur
        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
        # 3- find the edges by canny filter
        _, imgTh = cv2.threshold(imgGray, threshold1D, threshold2D, cv2.THRESH_BINARY)
        imgCanny = cv2.Canny(imgBlur, threshold1D, threshold1D)
        getContoursDW(imgTh)
        displayDW(imgContourDW)
        stack = stackImages(0.9, ([img, imgCanny], [imgContourDW, imgTh]))
        cv2.imshow('DownWardCamera Stacking ', stack)

        print('--------------------------------')
        # myDirectionDW = 1    #####################################################################################################################################################################################################################################
        if myDirectionDW == 1 and myDirectionDW_LR == 1:
            left_right_velocity = 12 ; for_back_velocity = 12
        elif myDirectionDW == 2 and myDirectionDW_LR == 2:
            left_right_velocity = -12 ; for_back_velocity = 12
        elif myDirectionDW == 3 and myDirectionDW_LR == 3:
            left_right_velocity = 12 ; for_back_velocity = -12
        elif myDirectionDW == 4 and myDirectionDW_LR == 4:
            left_right_velocity = -12 ; for_back_velocity = -12
        elif myDirectionDW == 1:  # GO LEFT
            left_right_velocity = -12 # 10
        elif myDirectionDW == 2:  # GO RIGHT
            left_right_velocity = 12
        elif myDirectionDW == 3:  # GO FORWARD
            for_back_velocity = 12
        elif myDirectionDW == 4:  # GO BACKWARD
            for_back_velocity = -12
        elif Landing_search == 0:
            if Search_right_side <= 11 :
                left_right_velocity = 40;Search_right_side = Search_right_side + 1 ; print('Search_right_side = ', Search_right_side)
            elif Search_left_side <= 11:
                left_right_velocity= -40; Search_left_side = Search_left_side + 1 ; print('Search_left_side = ', Search_left_side)
        else:
            left_right_velocity = 0
            for_back_velocity = 0
            up_down_velocity = 0
            yaw_velocity = 0
        if me.send_rc_control:
            me.send_rc_control(left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity)
            print('DW_Cam_Dir:', myDirectionDW)
        time.sleep(0.25)
        left_right_velocity = 0;    for_back_velocity = 0;  up_down_velocity = 0;   yaw_velocity = 0
        myDirectionDW = 0
        print('---')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#this function Get the coordinates from getContoursDW function   이 함수는 getContoursDW 함수에서 좌표를 가져옵니다.

####################################################################################
### MY MAIN PROGRAM ################################################################
startCounter = 0

me = tello.Tello()
me.connect()
print("---------------------------------")
print("Power: ", me.get_battery(), "%", ', Temperature: ', me.get_temperature(), ' oC')

LiveShowDW_TH1 = Thread(target=videoLiveShowDW)
LiveShowDW_TH2 = Thread(target=videoLiveShowDW)

sleepTime= 2
################# Flight
if startCounter == 0:
    me.takeoff();
    me.set_speed(50)
    me.move_forward(100)
    me.move_up(50)
    me.move_right(20)
    #me.move_forward(150)
    me.move_forward(135)
    # me.move_forward(130)
    me.move_down(60)
    me.streamoff() ; time.sleep(0.2)
    me.set_video_direction(me.CAMERA_DOWNWARD); time.sleep(0.4)
    me.streamon() ; time.sleep(0.3)

    # LiveShowDW_TH.start();time.sleep(4 * sleepTime)
    LiveShowDW_TH1.start();time.sleep(4 * sleepTime)
    LiveShowDW_TH2.start();time.sleep(4 * sleepTime);me.land() # for make it work in background: LiveShowDW_TH2.start()   백그라운드에서 작동하게 하려면: LiveShowDW_TH2.start()
    startCounter = 1
#Mission procedure 임무 절차

me.streamoff()
keepShowingDW = False ; LiveShowDW_TH1.join()
LiveShowDW_TH2.join()
cv2.destroyAllWindows()