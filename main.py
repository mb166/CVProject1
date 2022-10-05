import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
class RS3CV():
    def __init__(self):
        pass

    def templateMatch(self, reference, image):
        found = None
        maxVal = 0
        template = cv2.imread(reference, 0)
        template = cv2.Canny(template, 50, 200)
        w, h = template.shape[::-1]
        source = cv2.imread(image)
        graySource = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        #graySource = cv2.equalizeHist(graySource)
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            resized = imutils.resize(graySource, width = int(graySource.shape[1] * scale))
            r = graySource.shape[1] / float(resized.shape[1])
            if resized.shape[0] < h or resized.shape[1] < w:
                break
            edged = cv2.Canny(resized, 50, 200)
            res = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, newMaxVal, _, maxLoc) = cv2.minMaxLoc(res)
            if maxVal == 0 or newMaxVal > maxVal:
                found = (res, r)
                maxVal = newMaxVal

        self.drawResults(found[0], w, h, source, found[1])
        
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::1]):
             cv2.rectangle(source, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
        
        #cv2.imshow(source)


    def drawResults(self, res, w, h, source, r):
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        topLeft = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        bottomRight = (int((topLeft[0] + w) * r), int((topLeft[1] + h) * r))
        cv2.rectangle(source,topLeft, bottomRight, 255, 2)
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(source,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle('Results')
        plt.show()

    def cascadeDetection(self, image):
        pass

    def featureMatchORB(self, reference, image):
        template = cv2.imread(reference, cv2.IMREAD_GRAYSCALE)
        source = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(template,None)
        kp2, des2 = orb.detectAndCompute(source,None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        
        img3 = cv2.drawMatches(template,kp1,source,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3),plt.show()

    def flannMatcher(self, reference, image):
        template = cv2.imread(reference, cv2.IMREAD_GRAYSCALE)
        source = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(template,None)
        kp2, des2 = sift.detectAndCompute(source,None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = cv2.DrawMatchesFlags_DEFAULT)
        img3 = cv2.drawMatchesKnn(template,kp1,source,kp2,matches,None,**draw_params)
        plt.imshow(img3,),plt.show()



RS3ObjectFinder = RS3CV()

RS3ObjectFinder.templateMatch('templates\Iron_rock.png','TestImages\Test2.png')
RS3ObjectFinder.templateMatch('templates\ironrock2.png','TestImages\Test2.png')
RS3ObjectFinder.templateMatch('templates\ironrock1.png','TestImages\Test2.png')
RS3ObjectFinder.templateMatch('templates\ironrock3.png','TestImages\Test2.png')

RS3ObjectFinder.featureMatchORB('templates\Iron_rock.png','TestImages\Test2.png')
RS3ObjectFinder.featureMatchORB('templates\ironrock2.png','TestImages\Test2.png')
RS3ObjectFinder.featureMatchORB('templates\ironrock1.png','TestImages\Test2.png')
RS3ObjectFinder.featureMatchORB('templates\ironrock3.png','TestImages\Test2.png')

RS3ObjectFinder.flannMatcher('templates\ironrock2.png','TestImages\Test2.png')
RS3ObjectFinder.flannMatcher('templates\ironrock1.png','TestImages\Test2.png')
RS3ObjectFinder.flannMatcher('templates\ironrock3.png','TestImages\Test2.png')

RS3ObjectFinder.flannMatcher('templates\ironrock2.png','TestImages\Test1.png')
RS3ObjectFinder.flannMatcher('templates\ironrock1.png','TestImages\Test1.png')
RS3ObjectFinder.flannMatcher('templates\ironrock3.png','TestImages\Test1.png')

