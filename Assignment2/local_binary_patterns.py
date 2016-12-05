# import the necessary packages
import numpy as np
 
class LocalBinaryPatterns:
  def __init__(self):
    pass
 
  def compute(self, X, keypoints):
    X = np.asarray(X)
    X = (1 << 7) * (X[0:-2, 0:-2] >= X[1:-1, 1:-1]) \
        + (1 << 6) * (X[0:-2, 1:-1] >= X[1:-1, 1:-1]) \
        + (1 << 5) * (X[0:-2, 2:] >= X[1:-1, 1:-1]) \
        + (1 << 4) * (X[1:-1, 2:] >= X[1:-1, 1:-1]) \
        + (1 << 3) * (X[2:, 2:] >= X[1:-1, 1:-1]) \
        + (1 << 2) * (X[2:, 1:-1] >= X[1:-1, 1:-1]) \
        + (1 << 1) * (X[2:, :-2] >= X[1:-1, 1:-1]) \
        + (1 << 0) * (X[1:-1, :-2] >= X[1:-1, 1:-1])
    res = []
    for x in keypoints:
      rows, cols = int(x.pt[1]), int(x.pt[0])
      if rows >= X.shape[0] or cols >= X.shape[1]:
        continue
      res.append(X[rows][cols])
    res = np.array(res)
    res = np.reshape(res, (-1, 1))
    return keypoints, np.uint8(res)