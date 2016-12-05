import cv2
import numpy as np
from local_binary_patterns import LocalBinaryPatterns

class MatcherHelper:
	def __init__(self):
		self.options = {
			'harris' : self.performHarrisDetector,
			'blob' : self.performBlobDetector,
			'dog' : self.performSIFTDetector,
			'm' : self.performMatching,
		}

	def performHarrisDetector(self, argument):
		image = cv2.imread(argument[2])
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

		gray = np.float32(gray)
		dst = cv2.cornerHarris(gray, 2, 3, 0.04)
		image[dst>0.01*dst.max()]=[0,0,255]
		cv2.imshow('harris', image)

	def performBlobDetector(self, argument):
		image = cv2.imread(argument[2], cv2.IMREAD_GRAYSCALE)
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

		# Detect blobs.
		keypoints = detector.detect(image)
		imageWithKeypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		 
		# Show keypoints
		cv2.imshow("Blob", imageWithKeypoints)

	def performSIFTDetector(self, argument):
		image = cv2.imread(argument[2], cv2.IMREAD_GRAYSCALE)
		sift = cv2.xfeatures2d.SIFT_create()
		keypoints = sift.detect(image, None)
	     
		imageWithKeypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv2.imshow('SIFT', imageWithKeypoints)

	def performMatching(self, argument):
		image1 = cv2.imread(argument[4], cv2.IMREAD_GRAYSCALE)
		image2 = cv2.imread(argument[5], cv2.IMREAD_GRAYSCALE)

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

		# Apply ratio test
		good = []
		for m,n in matches:
		    if m.distance < 0.75 * n.distance:
		        good.append(m)

		# cv2.drawMatchesKnn expects list of lists as matches.
		image3 = np.zeros(image1.shape)
		image3 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good, image3)
		# image3 = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], flags=2)

		cv2.imshow('Matches', image3)

