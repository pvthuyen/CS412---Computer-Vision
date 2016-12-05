import sys
import cv2
from matcher_helper import MatcherHelper

helperInstance = MatcherHelper()

helperInstance.options[sys.argv[1]](sys.argv)

cv2.waitKey(0)
cv2.destroyAllWindows()
