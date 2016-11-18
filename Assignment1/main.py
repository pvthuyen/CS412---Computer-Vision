import sys
import cv2
from utilities import Utilities

if len(sys.argv) < 2:
	cap = cv2.VideoCapture(0)
else:
	originalImage = cv2.imread(sys.argv[1])

utilitiesInstance = Utilities()

while(True):
	if len(sys.argv) < 2:
	  # Capture frame-by-frame
	  ret, frame = cap.read()
	else:
		frame = originalImage

	# print frame.max(), frame.min()

	rawKeyPressed = cv2.waitKey(1)

	print rawKeyPressed

	if rawKeyPressed == ord('q'):
		break

	frame = utilitiesInstance.processFrame(frame, rawKeyPressed)

	# print frame.max(), frame.min(), frame.shape

	# Display the resulting frame
	cv2.imshow('frame', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
