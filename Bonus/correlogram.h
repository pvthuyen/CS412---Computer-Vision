/*
	Created by pvthuyen
	This header contains computeCorrelogram to compute correlogram of a grayscale image.
	computeCorrelogram takes an image and the distance range as input and returns a vector
	of double containing the probability of two pixels with distance c having the same color.
*/
#ifndef CORRELOGRAM_H
#define CORRELOGRAM_H

#include <opencv2/core/core.hpp>

using namespace cv;

vector <double> computeCorrelogram(const Mat &image, int distanceRange) {
	int pixelNum[300];
	int correlogram[300][distanceRange + 5];

	for (int pi = 0; pi < image.rows; pi++)
	{
		for (int pj = 0; pj < image.cols; pj++)
		{
			int color = image.at<uchar>(pi, pj);
			pixelNum[color]++;

			for (int d = 1; d <= distanceRange; d++)
			{
				int sameColorNum=0; // number of pixels with same color which have distance d
				int totalBoundaryNum=0; // total number of pixels which have distance d

				// upper margin
				for (int i = pi - d, j = pj - d; j <= pj + d; j++)
				{
					if (i < 0)
						break;
					if (j < 0 || j >= image.cols)
						continue;
					int otherColor = image.at<uchar>(i, j);
					if (color == otherColor)
					{
						sameColorNum++;
					}
					totalBoundaryNum++;
				}

				// lower margin
				for (int i = pi + d, j = pj - d; j <= pj + d; j++)
				{
					if (i >= image.rows)
						break;
					if (j < 0 || j >= image.cols)
						continue;
					int otherColor = image.at<uchar>(i, j);
					if (color == otherColor)
					{
						sameColorNum++;
					}
					totalBoundaryNum++;
				}

				// left margin
				for (int i = pi - d + 1, j = pj - d; i <= pi + d - 1; i++)
				{
					if (j < 0)
						break;
					if (i < 0 || i >= image.rows)
						continue;
					int otherColor = image.at<uchar>(i, j);
					if (color == otherColor)
					{
						sameColorNum++;
					}
					totalBoundaryNum++;
				}

				// right margin
				for (int i = pi - d + 1, j = pj + d; i <= pi + d - 1; i++)
				{
					if (j >= image.cols)
						break;
					if (i < 0 || i >= image.rows)
						continue;
					int otherColor = image.at<uchar>(i, j);
					if (color == otherColor)
					{
						sameColorNum++;
					}
					totalBoundaryNum++;
				}
				if (totalBoundaryNum != 0)
				{
					correlogram[color][d - 1] += (double)sameColorNum / (double)totalBoundaryNum;
				}
			}
		}
	}

	vector <double> res;
	for(int c = 0; c < 256; c++)
	{
		for(int d = 0; d < distanceRange; d++)
		{
			if (pixelNum[c] != 0)
				res.push_back(correlogram[c][d] / (double)pixelNum[c]);
			else
				res.push_back(0);
		}
	}
	return res;
}

#endif // CORRELOGRAM_H