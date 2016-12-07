import sys
import cv2
from matcher_helper import MatcherHelper

helperInstance = MatcherHelper()

helperInstance.process(sys.argv)
