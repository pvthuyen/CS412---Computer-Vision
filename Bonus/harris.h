#ifndef HARRIS_H
#define HARRIS_H

#include <opencv2/opencv.hpp>

#include <vector>

using namespace cv;
using namespace std;

vector<KeyPoint> harrisCorner(const Mat &image, int apertureSize = 3, float alpha = 0.01, float quality = 0.04) {
	// (1) Convert to greyscalescale image
	Mat grayscale;
	cvtColor(image, grayscale, CV_BGR2GRAY);

	// (2) Compute Derivatives
	Mat gradX, gradY;
	Sobel(grayscale, gradX, CV_32F, 1, 0);
	Sobel(grayscale, gradY, CV_32F, 0, 1);

	Mat gradX2 = gradX.mul(gradY);
	Mat gradY2 = gradY.mul(gradY);
	Mat gradXY = gradX.mul(gradY);

	// (3) Gaussian Filtering
	Mat filteredGradX2, filteredGradY2, filteredGradXY;
	GaussianBlur(gradX2, filteredGradX2, Size(apertureSize, apertureSize), apertureSize / 6.0);
	GaussianBlur(gradY2, filteredGradY2, Size(apertureSize, apertureSize), apertureSize / 6.0);
	GaussianBlur(gradXY, filteredGradXY, Size(apertureSize, apertureSize), apertureSize / 6.0);

  	// Compute harris responses
	Mat temp = filteredGradX2.mul(filteredGradY2);
	Mat determinants = temp - filteredGradXY;
	temp = filteredGradX2 + filteredGradY2;
	Mat traces = temp.mul(temp);

	Mat responses = determinants - alpha * traces;

  	// Detect corner based on response values
	float maxResponse = -1e9;
	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			maxResponse = max(maxResponse, responses.at<float>(i, j));
		}
	}

	vector<KeyPoint> corners;
	for (int i = 0; i < responses.rows; ++i) {
		for (int j = 0; j < responses.cols; ++j) {
			if (responses.at<float>(i, j) >= maxResponse * quality) {
				corners.push_back(KeyPoint(j, i, 1));
			}
		}
	}
	return corners;
}


#endif // HARRIS_H