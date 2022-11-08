import numpy as np
import win32gui, win32ui, win32con
import cv2 as cv


gameWindow = None
width = 0
height = 0
croppedX = 0
croppedY = 0
offsetX = 0
offsetY = 0
def WindowCapture(windowName=None):
        #getting handle for window
        global gameWindow
        global width
        global height
        global croppedX
        global croppedY
        global offsetX
        global offsetY
        #getting handle for window
        if windowName is None:
            gameWindow = win32gui.GetDesktopWindow()
        else:
            gameWindow = win32gui.FindWindow(None, windowName)
            if not gameWindow:
                raise Exception('Window not found: {}'.format(windowName))

        #get the window size
        windowRectangle = win32gui.GetWindowRect(gameWindow)
        width = windowRectangle[2] - windowRectangle[0]
        height = windowRectangle[3] - windowRectangle[1]

        #account for the window border and titlebar and cut them off
        borderPixels = 8
        titlebarPixels = 30
        width = width - (borderPixels * 2)
        height = height - titlebarPixels - borderPixels
        croppedX = borderPixels
        croppedY = titlebarPixels

        offsetX = windowRectangle[0] + croppedX
        offsetY = windowRectangle[1] + croppedY

def drawRectangles(screenshot, rectangles):
    #BGR
    lineColor = (0, 255, 0)
    lineType = cv.LINE_4

    for (x, y, w, h) in rectangles:
        #determining rectangle positinos
        topLeft = (x, y)
        bottomRight = (x + w, y + h)
        #drawing rectangles
        cv.rectangle(screenshot, topLeft, bottomRight, lineColor, lineType=lineType)

    return screenshot

def getScreenshot():

        #getting window image data
        wDC = win32gui.GetWindowDC(gameWindow)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (width, height), dcObj, (croppedX, croppedY), win32con.SRCCOPY)

        #have to convert to format opencv likes
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)

        #free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(gameWindow, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        return img


#capturing the game window
WindowCapture('RuneScape')

#load the trained model
ironRockClassifier = cv.CascadeClassifier('cascade/cascade.xml')

#event loop that will display a live feed of the game window and do object detection
while(True):
    #get recent frame of game
    screenshot = getScreenshot()

    #detect objects
    rectangles = ironRockClassifier.detectMultiScale(screenshot)

    #draw rectangles on screenshot
    detectionImage = drawRectangles(screenshot, rectangles)

    #display the images
    cv.imshow('Detected Objects', detectionImage)

    #pressing q quits the loop and closes program
    key = cv.waitKey(1)
    if key == ord('q'):
        cv.destroyAllWindows()
        break