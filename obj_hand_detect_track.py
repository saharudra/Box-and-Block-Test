import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# Function to find angle between two vectors
def Angle(v1,v2):
 dot = np.dot(v1,v2)
 x_modulus = np.sqrt((v1*v1).sum())
 y_modulus = np.sqrt((v2*v2).sum())
 cos_angle = dot / x_modulus / y_modulus
 angle = np.degrees(np.arccos(cos_angle))
 return angle
# Function to find distance between two points in a list of lists
def FindDistance(A,B):
 return np.sqrt(np.power((A[0][0]-B[0][0]),2) + np.power((A[0][1]-B[0][1]),2))

# open camera object

cap = cv2.VideoCapture('sat9.mp4');
fourcc = cv2.cv.CV_FOURCC(*'MJPG')
video = cv2.VideoWriter('hand_object_tracking.avi', fourcc, 30.0, (854, 640));

# Decrease frame size

cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,1000)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,600)
i =0
no = 0
# total = 0
# red = 0
xstore=[]
ystore=[]
frameno=[]
j=0

while cap.isOpened():
    # Take each frame

    ret, frame = cap.read()
    if ret:
        cropped = frame[:600, :400]
        print(np.shape(frame))
        # cv2.imshow("Cropped", cropped)

        ####################################################################################################################
        ######################################OBJECT DETECTION##############################################################
        # Convert BGR to HSV

        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

        # Red Mask
        # define range of red color in HSV
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([6, 255, 255])

        redMask = cv2.inRange(hsv,lower_red,upper_red)
        # cv2.imshow("RedMask", redMask)

        # Blue Mask

        # define range of blue color in HSV
        lower_blue = np.array([105, 70, 50])
        upper_blue = np.array([135, 255, 255])
        blueMask = cv2.inRange(hsv, lower_blue, upper_blue)
        # cv2.imshow("blueMask", blueMask)
        # Yellow Mask

        # define range of yellow color in HSV
        lower_yellow = np.array([28, 100, 100])
        upper_yellow = np.array([36, 255, 255])

        yellowMask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # cv2.imshow("yellowMask", yellowMask)

        # Green Mask

        # define range of green color in HSV
        lower_green = np.array([40, 100, 30])
        upper_green = np.array([100, 255, 255])

        greenMask = cv2.inRange(hsv, lower_green, upper_green)
        # cv2.imshow("greenMask", greenMask)

        # Contour Code

        redContours, redHierarchy = cv2.findContours(redMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        greenContours, greenHierarchy = cv2.findContours(greenMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        blueContours, blueHierarchy = cv2.findContours(blueMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        yellowContours, yellowHierarchy = cv2.findContours(yellowMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        blueTotal = 0
        greenTotal = 0
        redTotal = 0
        yellowTotal = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.drawContours(frame,redContours, -1, (0, 100, 100), 3)


        # loop over the red contours
        for c in redContours:
            # approximate the contour
            redPeri = cv2.arcLength(c, True)
            redApprox = cv2.approxPolyDP(c, 0.08*redPeri, True)
            redArea = cv2.contourArea(c)
            if redArea > 100:
                #cv2.drawContours(frame, [redApprox],-1,(255, 255,255),4)
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                redTotal = redTotal + 1
            # if(redTotal > red):
            #     red = redTotal

        # loop over the blue contours
        for c in blueContours:
            # approximate the contour
            bluePeri = cv2.arcLength(c, True)
            blueApprox = cv2.approxPolyDP(c, 0.08* bluePeri, True)
            blueArea=cv2.contourArea(c)
            if blueArea > 300:
                #cv2.drawContours(frame, [blueApprox], -1, (255, 255, 0), 4)
                x, y, w, h = cv2.boundingRect(c)
                w = 32
                h = 32
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                blueTotal = blueTotal + 1

        # loop over the green contours
        for c in greenContours:
            # approximate the contour
            greenPeri = cv2.arcLength(c, True)
            greenApprox = cv2.approxPolyDP(c, 0.08 * greenPeri, True)
            greenArea = cv2.contourArea(c)
            currGreenTotal = greenTotal
            if greenArea > 250:
                #cv2.drawContours(frame, [greenApprox], -1, (0, 255, 0), 4)
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                greenTotal = 1 + greenTotal

        # loop over the yellow contours
        for c in yellowContours:
            # approximate the contour
            yellowPeri = cv2.arcLength(c, True)
            yellowApprox = cv2.approxPolyDP(c, 0.08 * yellowPeri, True)
            yellowArea = cv2.contourArea(c)
            if yellowArea > 300:
                #cv2.drawContours(frame, [yellowApprox], -1, (255, 0, 255), 4)
                x, y, w, h = cv2.boundingRect(c)
                w = 32
                h = 32
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0, 255), 2)
                yellowTotal = yellowTotal + 1

        total = blueTotal + greenTotal + yellowTotal + redTotal
        # if(currTotal > total):
        #     total = currTotal
        print('Total Number of blocks: ' + str(total))
        cv2.putText(frame, "num total " + str(total), (50, 40), font, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "num red " + str(redTotal), (50, 80), font, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "num blue " + str(blueTotal), (50, 120), font, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "num yellow " + str(yellowTotal), (50, 160), font, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "num green " + str(greenTotal), (50, 200), font, 0.8, (255, 255, 255), 2)

        #cv2.imshow('frame', frame)
        #cv2.waitKey(20)



    ########################################################################################################################
    ######################################HAND TRACKING#####################################################################


        # Blur the image
        blur = cv2.blur(frame,(3,3))

        # Convert to HSV color space
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

        # Create a binary image with where white will be skin colors and rest is black
        mask2 = cv2.inRange(hsv,np.array([6,50,50]),np.array([15,230,100]))

        # Kernel matrices for morphological transformation
        kernel_square = np.ones((11,11),np.uint8)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

        # Perform morphological transformations to filter out the background noise
        # Dilation increase skin color area
        # Erosion increase skin color area
        dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
        erosion = cv2.erode(dilation,kernel_square,iterations = 1)
        dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
        filtered = cv2.medianBlur(dilation2,5)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
        dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        median = cv2.medianBlur(dilation2,5)
        ret,thresh = cv2.threshold(median,127,255,0)

        # Find contours of the filtered frame
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # Draw Contours
        # cv2.drawContours(frame, cnt, -1, (122,122,0), 3)
        # cv2.imshow('Dilation',median)

        # Find Max contour area (Assume that hand is in the frame)
        max_area=100
        ci=0
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if(area>max_area):
                max_area=area
                ci=i

        # Largest area contour
        cnts = contours[ci]

        # Find convex hull
        hull = cv2.convexHull(cnts)

        # Find convex defects
        hull2 = cv2.convexHull(cnts,returnPoints = False)
        defects = cv2.convexityDefects(cnts,hull2)

        # Get defect points and draw them in the original image
        FarDefect = []
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnts[s][0])
            end = tuple(cnts[e][0])
            far = tuple(cnts[f][0])
            FarDefect.append(far)
            cv2.line(frame,start,end,[0,255,0],1)
            # cv2.circle(frame,far,10,[100,255,255],3)

        # Find moments of the largest contour
        moments = cv2.moments(cnts)

        # Central mass of first order moments
        if moments['m00']!=0:
            cx = int(moments['m10']/moments['m00']) # cx = M10/M00
            cy = int(moments['m01']/moments['m00']) # cy = M01/M00
        centerMass=(cx,cy)
        xstore.append(cx)
        ystore.append(cy)
        j = j + 1
        frameno.append(j)

        # Draw center mass
        cv2.circle(frame,centerMass,7,[100,0,255],2)
        cv2.putText(frame,'Center',tuple(centerMass),font,2,(255,255,255),2)

        # Distance from each finger defect(finger webbing) to the center mass
        distanceBetweenDefectsToCenter = []
        for i in range(0,len(FarDefect)):
            x =  np.array(FarDefect[i])
            centerMass = np.array(centerMass)
            distance = np.sqrt(np.power(x[0]-centerMass[0],2)+np.power(x[1]-centerMass[1],2))
            distanceBetweenDefectsToCenter.append(distance)

        # Get an average of three shortest distances from finger webbing to center mass
        sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
        AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])

        # Get fingertip points from contour hull
        # If points are in proximity of 80 pixels, consider as a single point in the group
        finger = []
        for i in range(0,len(hull)-1):
            if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
                if hull[i][0][1] <500 :
                    finger.append(hull[i][0])

        # The fingertip points are 5 hull points with largest y coordinates
        finger =  sorted(finger,key=lambda x: x[1])
        fingers = finger[0:5]

        # Calculate distance of each finger tip to the center mass
        fingerDistance = []
        for i in range(0,len(fingers)):
            distance = np.sqrt(np.power(fingers[i][0]-centerMass[0],2)+np.power(fingers[i][1]-centerMass[0],2))
            fingerDistance.append(distance)

        # Finger is pointed/raised if the distance of between fingertip to the center mass is larger
        # than the distance of average finger webbing to center mass by 130 pixels
        result = 0
        for i in range(0,len(fingers)):
            if fingerDistance[i] > AverageDefectDistance+130:
                result = result +1

        # Print number of pointed fingers
        # cv2.putText(frame,str(result),(100,100),font,2,(255,255,255),2)

        # Print bounding rectangle
        x,y,w,h = cv2.boundingRect(cnts)
        img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.drawContours(frame,[hull],-1,(255,255,255),2)

        ##### Show final image ########
        video.write(frame)
        cv2.imshow('Detect&Track',frame)
        ###############################

        # Print execution time
        # print time.time()-start_time

        # close the output video by pressing 'ESC'
        k = cv2.waitKey(20) & 0xFF

        if k == 27:
            break
    else:
        break

############################################## PLOTTING FUNCTION #######################################################
rate = 30
frameno2 = [i / rate for i in frameno]
xstore2=map(int,xstore)
ystore2=map(int,ystore)
x_mean =sum(xstore2) / float(len(xstore2))
y_mean =sum(ystore2) / float(len(ystore2))

x_mean2 = [np.mean(xstore) - x_mean for i in frameno]
y_mean2 = [np.mean(ystore) - y_mean for i in frameno]
xstore3=[x - x_mean for x in xstore2]
ystore3=[x - y_mean for x in ystore2]
fig, ax = plt.subplots()
# Plot the data
data_line = ax.plot(frameno2, xstore3, label='Deviation of pixels in x direction from mean'+str(round(x_mean)))
# Plot the average line
mean_line = ax.plot(frameno2, x_mean2, label='Mean-x', linestyle='--')
# Make a legend
legend = ax.legend(loc='lower right')
plt.xlabel('time(sec)', fontsize=18)
plt.ylabel('Deviation(No of Pixels)', fontsize=16)
fig2, ax2 = plt.subplots()
# Plot the data
data_line2 = ax2.plot(frameno2, ystore3, label='Deviation of pixels in y direction from mean '+str(round(y_mean)))
# Plot the average line
mean_line2 = ax2.plot(frameno2, y_mean2, label='Mean-y', linestyle='--')
# Make a legend
legend2 = ax2.legend(loc='lower right')
plt.xlabel('time(sec)', fontsize=18)
plt.ylabel('Deviation(No of pixels)', fontsize=16)
plt.draw()
plt.pause(15)
plt.close('all')

cap.release()
video.release()
cv2.destroyAllWindows()
