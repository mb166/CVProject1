import cv2 as cv
import numpy as np
import os
from time import time
from windowcapture import WindowCapture
from detection import Detection
from vision import Vision
from bot import RSBot, BotState


os.chdir(os.path.dirname(os.path.abspath(__file__)))


DEBUG = False

# initialize the WindowCapture class
wincap = WindowCapture('Runescape')
# load the detector
treeDetector = Detection('treeCascade\cascade.xml')
bankDetector = Detection('chestBankCascade\cascade.xml')
# load an empty Vision class
vision = Vision()
# initialize the bot
bot = RSBot((wincap.offset_x, wincap.offset_y), (wincap.w, wincap.h))

wincap.start()
treeDetector.start()
bankDetector.start()
bot.start()


while(True):

    # if we don't have a screenshot yet, don't run the code below this point yet
    if wincap.screenshot is None:
        continue

    # give detector the current screenshot to search for objects in
    treeDetector.update(wincap.screenshot)
    bankDetector.update(wincap.screenshot)

    # Serve bot needed data based on state
    if bot.state == BotState.INITIALIZING:
        # initializing procedure
        targets = vision.get_click_points(treeDetector.rectangles)
        bot.update_targets(targets)
    elif bot.state == BotState.SEARCHING:
        # get points to click and verify they are the object we want
        targets = vision.get_click_points(treeDetector.rectangles)
        bot.update_targets(targets)
        bot.update_screenshot(wincap.screenshot)
    elif bot.state == BotState.MOVING:
        # when moving, we need fresh screenshots to determine when we've stopped moving
        bot.update_screenshot(wincap.screenshot)
    elif bot.state == BotState.MINING:
        # just wait for resources to gather
        pass
    elif bot.state == BotState.BANKING:
        targets = vision.get_click_points(bankDetector.rectangles)
        bot.update_targets(targets)
        bot.update_screenshot(wincap.screenshot)

    if DEBUG:
        # draw the detection results onto the original image
        detection_image = vision.draw_rectangles(wincap.screenshot, treeDetector.rectangles)
        # display the images
        cv.imshow('Matches', detection_image)

    # press q to exit.
    key = cv.waitKey(1)
    if key == ord('q'):
        wincap.stop()
        treeDetector.stop()
        bankDetector.stop()
        bot.stop()
        cv.destroyAllWindows()
        break

print('Done.')
