#include "harris.h"
#include "correlogram.h"
#include "canny.h"
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv) {
	if (argc < 2) {
		return 0;
	}

	Mat image = imread(argv[1], 0);
	
	// // Harris
 //    vector<KeyPoint> corners = harrisCorner(image);
 //    Mat cornerImage;
 //    drawKeypoints(image, corners, cornerImage);
 //    imshow("Harris Corners", cornerImage);

    // Correlogram
    vector <double> correlogram = computeCorrelogram(image, 10);

    for (int i = 0; i < correlogram.size(); ++i)
    	cout << correlogram[i] << " ";
    cout << endl;

    // Canny
    Mat cannyOutput = cannyDetector(image);
    imshow("Canny Edges", cannyOutput);

    waitKey(0);
	return 0;
}