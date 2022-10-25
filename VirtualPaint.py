import cv2
import os
import numpy as np
import handTrackingModule
import pytesseract


def main():
    brushThickness = 10
    eraserThickness = 100
    counter = "Selection mode"
    printText = 0
    folderPath = "header"
    myList = os.listdir(folderPath)
    overLayList = []

    for i in range(0, len(myList)):
        image = cv2.imread(f'{folderPath}/{myList[i]}')
        overLayList.append(image)
    print(len(overLayList))
    #overlay one of the images by default
    header = overLayList[0]
    drawColor = (0, 0, 255)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = handTrackingModule.handDetector(detectionCon=0.85)

    xp, yp = 0, 0

    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    while True:
        #import the image
        success, img = cap.read()
        img = cv2.flip(img, 1)

        #find hand landmarks
        img = detector.findHands(img) #this will draw image and detect the hand
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:

            #print(lmList)

            #find thew tip point of middle finger and tip finger
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            #check which fingers are up

            fingers = detector.fingersUp()

            #if selection mode(when two fingers are up) then we select
            if fingers[1] and fingers[2] == True:
                xp, yp = 0, 0
                cv2.putText(img, "Selection Mode", (20, 700), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
                #cv2.rectangle(img, (x1, y1-15), (x2, y2 + 25), drawColor, cv2.FILLED)
                if counter == "Selection mode":
                    counter = "Drawing mode"

                if y1 < 125:
                    if 0 < x1 < 200:
                        if printText == 0:
                            text = pytesseract.image_to_string(imgCanvas)
                            print(f"You wrote: {text}")
                            printText = 1
                    elif 250 < x1 < 450:
                        header = overLayList[0]
                        drawColor = (88, 88, 255)
                        printText = 0
                    elif 550 < x1 < 750:
                        header = overLayList[2]
                        drawColor = (250, 255, 88)
                        printText = 0
                    elif 800 < x1 < 950:
                        header = overLayList[1]
                        drawColor = (88, 255, 88)
                        printText = 0
                    elif 1050 < x2 < 1200:
                        header = overLayList[3]
                        drawColor = (0, 0, 0)
                        printText = 0

                #cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 25), drawColor, cv2.FILLED)


            #if drawing mode (index finger is up)
            elif fingers[1] and fingers[2] == False:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                cv2.putText(img, "Drawing Mode", (20, 700), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
                if counter == "Drawing mode":
                    counter = "Selection mode"

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if drawColor == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

                #keep updating the points
                xp, yp = x1, y1

        #covert into gray image
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)

        #convert into binary image and reverse
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        #overlay the image and setting the header image
        img[0:125, 0:1280] = header
        img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
        cv2.imshow("Image",img)
        #cv2.imshow("Canvas", imgCanvas)
        #cv2.imshow("Inv", imgInv)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Program terminated!')