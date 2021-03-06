import cv2
import numpy as np
from local_binary_patterns import LocalBinaryPatterns

class MatcherHelper:
    def __init__(self):
        self._options = {
            'harris' : self.performHarrisDetector,
            'blob' : self.performBlobDetector,
            'dog' : self.performSIFTDetector,
            'm' : self.performMatching,
            'h' : self.displayHelp,
        }
        self._helpText = [
            "If the image path is not specified, images will be taken from camera.",
            "You can press any button to capture the image"
            "harris image.jpg - detect key points using harris",
            "   algorithm and show the keypoints in original image.",
            "blob image.jpg - detect key points using blob algorithm",
            "   and show the keypoints in original image.",
            "dog image.jpg - detect key points using DoG Algorithm",
            "   and show keypoints in original image.",
            "m harris sift - image1.jpg image2.jpg match and show results of image1 and image2",
            "   using Harris detector and SIFT descriptor.",
            "m dog sift image1.jpg image2.jpg - match and show results of image1 and image2",
            "   using DoG detector and SIFT descriptor.",
            "m blob sift image1.jpg image2.jpg - match and show results of image1 and image2",
            "   using using Blob detector and SIFT descriptor.",
            "m harris lbp image1.jpg image2.jpg - match and show results of image1 and image2",
            "   using Harris detector and LBP descriptor.",
            "m dog lbp image1.jpg image2.jpg - match and show results of image1 and image",
            "   using DoG detector and LBP descriptor.",
            "m blob lbp image1.jpg image2.jpg - match and show results of image1 and image2",
            "   using Blob detector and LBP descriptor.",
            "h - display this help windows."
        ]

    def process(self, argument):
        if argument[1] == 'm':
            if len(argument) < 6:
                cap = cv2.VideoCapture(0)
                while True:
                    _, image1 = cap.read()
                    cv2.imshow("Image 1", image1)
                    if cv2.waitKey(1) != -1:
                        break

                while True:
                    _, image2 = cap.read()
                    cv2.imshow("Image 2", image2)
                    if cv2.waitKey(1) != -1:
                        break                        
                if not cap:
                    cap.release()
            else:
                image1 = cv2.imread(argument[4])
                image2 = cv2.imread(argument[5])
            self.performMatching(argument, image1, image2)
            cv2.waitKey(0)
        elif argument[1] != 'h':
            if len(argument) < 3:
                cap = cv2.VideoCapture(0)
                while True:
                    _, image = cap.read()
                    cv2.imshow("Camera", image)
                    if cv2.waitKey(1) != -1:
                        break
                if not cap:
                    cap.release()
            else:
                image = cv2.imread(argument[2])

            self._options[argument[1]](image)
        else:
            self.displayHelp()
            cv2.waitKey(0)

        cv2.destroyAllWindows()


    def performHarrisDetector(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(image)

        cv2.namedWindow("Trackbar")
        cv2.createTrackbar("Block size", "Trackbar", 0, 9, self.placeHolder)
        cv2.createTrackbar("Aperture size", "Trackbar", 0, 9, self.placeHolder)
        lastBlockSize = -1
        lastApertureSize = -1

        while True:
            blockSize = cv2.getTrackbarPos("Block size", "Trackbar") + 2
            apertureSize = cv2.getTrackbarPos("Aperture size", "Trackbar") * 2 + 3 
            if blockSize != lastBlockSize or apertureSize != lastApertureSize:      
                dst = cv2.cornerHarris(gray, blockSize, apertureSize, 0.04)
                keypoints = np.argwhere(dst>0.01*dst.max())
                keypoints = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints]
                imageWithKeypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow('harris', imageWithKeypoints)
                lastBlockSize = blockSize
                lastApertureSize = apertureSize
            if cv2.waitKey(1) != -1:
                break

    def performBlobDetector(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow("Trackbar")
        cv2.createTrackbar("Min Area", "Trackbar", 0, 100, self.placeHolder)
        cv2.createTrackbar("Max Area", "Trackbar", 100, 100, self.placeHolder)
        # cv2.createTrackbar("Min Circularity", "Trackbar", 0, 100, self.placeHolder)
        # cv2.createTrackbar("Max Circularity", "Trackbar", 100, 100, self.placeHolder)
        # cv2.createTrackbar("Min Convexity", "Trackbar", 0, 100, self.placeHolder)
        # cv2.createTrackbar("Max Convexity", "Trackbar", 100, 100, self.placeHolder)
        lastMinArea = -1
        lastMaxArea = -1

        while True:
            minArea = cv2.getTrackbarPos("Min Area", "Trackbar")
            maxArea = cv2.getTrackbarPos("Max Area", "Trackbar") * 100

            if minArea != lastMinArea or maxArea != lastMaxArea:
                # Setup SimpleBlobDetector parameters.
                params = cv2.SimpleBlobDetector_Params()

                params.filterByColor = True;
                params.blobColor = 255;
                 
                # Change thresholds
                params.minThreshold = 0;
                params.maxThreshold = 10000;
                 
                # Filter by Area.
                params.filterByArea = True
                params.minArea = minArea
                params.maxArea = maxArea
                 
                # Filter by Circularity
                params.filterByCircularity = True
                params.minCircularity = 0.1
                 
                # Filter by Convexity
                params.filterByConvexity = True
                params.minConvexity = 0.2
                 
                # Filter by Inertia
                params.filterByInertia = True
                params.minInertiaRatio = 0.01
                # Create a detector with the parameters
                detector = cv2.SimpleBlobDetector_create(params)

                # Detect blobs.
                keypoints = detector.detect(image)
                imageWithKeypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                 
                # Show keypoints
                cv2.imshow("Blob", imageWithKeypoints)

                lastMinArea = minArea
                lastMaxArea = maxArea
            if cv2.waitKey(1) != -1:
                break

    def performSIFTDetector(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow("Trackbar")
        cv2.createTrackbar("Octave layers", "Trackbar", 0, 9, self.placeHolder)
        cv2.createTrackbar("Contrast Threshold", "Trackbar", 0, 20, self.placeHolder)
        lastOctaveLayers = -1
        lastContastThreshold = -1

        while True:
            octaveLayers = cv2.getTrackbarPos("Octave layers", "Trackbar") + 3
            contrastThreshold = (cv2.getTrackbarPos("Contrast Threshold", "Trackbar") + 4) / 100.0
            if lastOctaveLayers != octaveLayers or contrastThreshold != lastContastThreshold:
                sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers = octaveLayers, contrastThreshold = contrastThreshold)
                keypoints = sift.detect(image, None)
             
                imageWithKeypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow('SIFT', imageWithKeypoints)

                lastOctaveLayers = octaveLayers
                lastContastThreshold = contrastThreshold
            if cv2.waitKey(1) != -1:
                break

    def performMatching(self, argument, image1, image2):
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        if argument[2] == 'harris':
            gray = np.float32(image1)
            dst = cv2.cornerHarris(gray, 2, 3, 0.04)
            keypoints1 = np.argwhere(dst>0.01*dst.max())
            keypoints1 = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints1]

            gray = np.float32(image2)
            dst = cv2.cornerHarris(gray, 2, 3, 0.04)
            keypoints2 = np.argwhere(dst>0.01*dst.max())
            keypoints2 = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints2]
        elif argument[2] == 'blob':
            # Setup SimpleBlobDetector parameters.
            params = cv2.SimpleBlobDetector_Params()

            params.filterByColor = True;
            params.blobColor = 255;
             
            # Change thresholds
            params.minThreshold = 0;
            params.maxThreshold = 10000;
             
            # Filter by Area.
            params.filterByArea = True
            params.minArea = 0
            params.maxArea = 100000
             
            # Filter by Circularity
            params.filterByCircularity = True
            params.minCircularity = 0.1
             
            # Filter by Convexity
            params.filterByConvexity = True
            params.minConvexity = 0.2
             
            # Filter by Inertia
            params.filterByInertia = True
            params.minInertiaRatio = 0.01
            # Create a detector with the parameters
            detector = cv2.SimpleBlobDetector_create(params)

            keypoints1 = detector.detect(image1)
            keypoints2 = detector.detect(image2)
        elif argument[2] == 'dog':
            sift = cv2.xfeatures2d.SIFT_create()
            keypoints1 = sift.detect(image1, None)
            keypoints2 = sift.detect(image2, None)

        if argument[3] == 'sift':
            extractor = cv2.xfeatures2d.SIFT_create()
        elif argument[3] == 'lbp':
            extractor = LocalBinaryPatterns()


        keypoints1, desc1 = extractor.compute(image1, keypoints1)
        keypoints2, desc2 = extractor.compute(image2, keypoints2)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)

        good = sorted(matches, key = lambda x: x[0].distance)
        good = good[:100]

        good = [x[0] for x in good]

        # cv2.drawMatchesKnn expects list of lists as matches.
        image3 = np.zeros(image1.shape)
        image3 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good, image3)
        # image3 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], flags=2)

        cv2.imshow('Matches', image3)

    def displayHelp(self):
        cv2.namedWindow("Help")
        helpFrame = np.zeros((500, 1280))
        originX = 25
        originY = 25

        for i in range(len(self._helpText)):
            cv2.putText(helpFrame, self._helpText[i], (originX, originY), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 255)   
            textSize = cv2.getTextSize(self._helpText[i], cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
            originY += textSize[1] + 10
        
        cv2.imshow("Help", helpFrame)

    def placeHolder(self, x):
        pass
