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
      rows, cols, size = int(x.pt[1] - 1), int(x.pt[0] - 1), int(x.size)
      rows_range = (max(0, rows - size), min(X.shape[0], rows + size + 1))
      cols_range = (max(0, cols - size), min(X.shape[1], cols + size + 1))
      window = X[rows_range[0]:rows_range[1], cols_range[0]:cols_range[1]]
      window = window.flatten()
      hist = np.histogram(window, bins = range(257))
      res.append(hist[0])

    res = np.array(res)
    res = np.reshape(res, (-1, 256))
    return keypoints, np.uint8(res)