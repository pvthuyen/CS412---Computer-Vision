import numpy as np
import cv2
import math

class Utilities:
	def __init__(self):
		self._channelIndex = 0
		self._options = {
			ord('i') : self.doNothing,
			ord('g') : self.convertToGrayscaleUsingOpenCV,
			ord('G') : self.convertToGrayscale,
			ord('c') : self.cycleThroughChannels,
			ord('s') : self.blurImageUsingOpenCV,
			ord('S') : self.blurImage,
			ord('x') : self.computeXDerivative,
			ord('y') : self.computeYDerivative,
			ord('m') : self.computeDerivative,
			ord('p') : self.drawGradient,
			ord('r') : self.rotateImage,
		}
		self._pressedKey = ord('i')
		self._windowsOn = False
		self._helpText = [
			"i - reload the original image (i.e. cancel any previous processing)",
			"w - save the current (possibly processed) image into the file out.jpg",
			"g - convert the image to grayscale using the openCV conversion function.",
			"G - convert the image to grayscale using your implementation of conversion function.",
			"c - cycle through the color channels of the image.",
			"s - convert the image to grayscale and smooth it using the openCV function.",
			"S - convert the image to grayscale and smooth it using my function.",
			"x - convert the image to grayscale and perform convolution with an x derivative filter.",
			"y - convert the image to grayscale and perform convolution with a y derivative filter.",
			"m - show the magnitude of the gradient normalized to the range [0,255].",
			"p - convert the image to grayscale and plot the gradient vectors of the image every N pixels.",
			"r - convert the image to grayscale and rotate it using an angle of Q degrees.",
			"h - Display this help window.",
		]

	def processFrame(self, frame, rawKeyPressed):
		if rawKeyPressed in self._options:
			self._pressedKey = rawKeyPressed
			if rawKeyPressed == ord('c'):
				self._channelIndex = (self._channelIndex + 2) % 3
			if self._windowsOn:
				cv2.destroyWindow("Trackbar")
				self._windowsOn = False

			frame = self._options[rawKeyPressed](frame)
		else:
			frame = self._options[self._pressedKey](frame)
			if rawKeyPressed == ord('w'):
				self.saveOutputImage(frame)
			if rawKeyPressed == ord('h'):
				self.displayHelp()

		return frame

	def doNothing(self, frame):
		return frame

	def saveOutputImage(self, frame):
		cv2.imwrite('out.jpg', frame)
		return frame

	def convertToGrayscaleUsingOpenCV(self, frame):
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		return frame

	def convertToGrayscale(self, frame):
		frame = 0.2126 * frame[:, :, 2] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 0]
		# frame = np.dstack((frame, frame, frame))
		return np.uint8(frame)

	def cycleThroughChannels(self, frame):
		result = np.zeros_like(frame)
		result[:, :, self._channelIndex] = frame[:, :, self._channelIndex]
		return result

	def placeHolder(self, x):
		pass

	def blurImageUsingOpenCV(self, frame):
		if not self._windowsOn:
			cv2.namedWindow("Trackbar");
			cv2.createTrackbar("Blur", "Trackbar", 0, 10, self.placeHolder)
			self._windowsOn = True

		kernelSize = cv2.getTrackbarPos("Blur", "Trackbar")
		kernelSize = 2 * kernelSize + 1
		frame = cv2.blur(frame, (kernelSize, kernelSize), 0)
		return frame

	def blurImage(self, frame):
		if not self._windowsOn:
			cv2.namedWindow("Trackbar");
			cv2.createTrackbar("Blur", "Trackbar", 0, 10, self.placeHolder)
			self._windowsOn = True

		kernelSize = cv2.getTrackbarPos("Blur", "Trackbar")
		kernelSize = 2 * kernelSize + 1
		kernel = np.ones((kernelSize, kernelSize))
		kernel = 1.0 / (kernelSize * kernelSize) * kernel
		frame = cv2.filter2D(frame, -1, kernel)
		return frame

	def computeXDerivative(self, frame):
		frame = cv2.GaussianBlur(frame, (3, 3), 0)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		xDerivativeKernel = np.zeros((3, 3))
		xDerivativeKernel[-1:0, 0] = -1
		xDerivativeKernel[1, 0] = -2
		xDerivativeKernel[-1:0, 2] = 1
		xDerivativeKernel[1, 2] = 2
		frame = cv2.filter2D(frame, -1, xDerivativeKernel)
		frame = 1.0 * (frame - frame.min()) / (frame.max() - frame.min()) * 255
		return np.uint8(frame)

	def computeYDerivative(self, frame):
		frame = cv2.GaussianBlur(frame, (3, 3), 0)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		yDerivativeKernel = np.zeros((3, 3))
		yDerivativeKernel[0, -1:0] = -1
		yDerivativeKernel[0, 1] = -2
		yDerivativeKernel[2, -1:0] = 1
		yDerivativeKernel[2, 1] = 2
		frame = cv2.filter2D(frame, -1, yDerivativeKernel)
		frame = 1.0 * (frame - frame.min()) / (frame.max() - frame.min()) * 255
		return np.uint8(frame)

	def computeDerivative(self, frame):
		frame = cv2.GaussianBlur(frame, (3, 3), 0)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		xDerivativeKernel = np.zeros((3, 3))
		xDerivativeKernel[-1:0, 0] = -1
		xDerivativeKernel[1, 0] = -2
		xDerivativeKernel[-1:0, 2] = 1
		xDerivativeKernel[1, 2] = 2
		xDerivative = cv2.filter2D(frame, -1, xDerivativeKernel)

		yDerivativeKernel = np.zeros((3, 3))
		yDerivativeKernel[0, -1:0] = -1
		yDerivativeKernel[0, 1] = -2
		yDerivativeKernel[2, -1:0] = 1
		yDerivativeKernel[2, 1] = 2
		yDerivative = cv2.filter2D(frame, -1, yDerivativeKernel)

		frame = 0.5 * np.abs(xDerivative) + 0.5 * np.abs(yDerivative)
		frame = 1.0 * (frame - frame.min()) / (frame.max() - frame.min()) * 255
		return np.uint8(frame)

	def drawGradient(self, frame):
		if not self._windowsOn:
			cv2.namedWindow("Trackbar");
			cv2.createTrackbar("Step", "Trackbar", 0, 10, self.placeHolder)
			self._windowsOn = True

		frame = cv2.GaussianBlur(frame, (3, 3), 0)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		xDerivativeKernel = np.zeros((3, 3))
		xDerivativeKernel[-1:0, 0] = -1
		xDerivativeKernel[1, 0] = -2
		xDerivativeKernel[-1:0, 2] = 1
		xDerivativeKernel[1, 2] = 2
		xDerivative = cv2.filter2D(frame, -1, xDerivativeKernel)

		yDerivativeKernel = np.zeros((3, 3))
		yDerivativeKernel[0, -1:0] = -1
		yDerivativeKernel[0, 1] = -2
		yDerivativeKernel[2, -1:0] = 1
		yDerivativeKernel[2, 1] = 2
		yDerivative = cv2.filter2D(frame, -1, yDerivativeKernel)

		stepSize = cv2.getTrackbarPos("Step", "Trackbar") * 5 + 20

		for x in range(stepSize / 2, frame.shape[0], stepSize):
			for y in range(stepSize / 2, frame.shape[1], stepSize):
				gradientAngle = math.atan2(yDerivative[x, y], xDerivative[x, y])
				dstX = int(x + stepSize * math.cos(gradientAngle))
				dstY = int(y + stepSize * math.sin(gradientAngle))
				cv2.arrowedLine(frame, (y, x), (dstY, dstX), (0, 0, 0))

		return np.uint8(frame)		

	def rotateImage(self, frame):
		if not self._windowsOn:
			cv2.namedWindow("Trackbar");
			cv2.createTrackbar("Rotation", "Trackbar", 0, 360, self.placeHolder)
			self._windowsOn = True

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		angle = cv2.getTrackbarPos("Rotation", "Trackbar")
		rows,cols = frame.shape

		M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
		frame = cv2.warpAffine(frame, M, (cols,rows))
		return frame

	def displayHelp(self):
		cv2.namedWindow("Help")
		helpFrame = np.zeros((250, 1280))
		originX = 25
		originY = 25

		for i in range(len(self._helpText)):
			cv2.putText(helpFrame, self._helpText[i], (originX, originY), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 255)	
			textSize = cv2.getTextSize(self._helpText[i], cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
			originY += textSize[1] + 10
		
		cv2.imshow("Help", helpFrame)
