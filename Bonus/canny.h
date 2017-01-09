#ifndef CANNY_H
#define CANNY_H

Mat cannyDetector(const Mat &image, double minVal = 50, double maxVal = 100, int apertureSize = 5) {
    // Filter out any noise.
    Mat filteredImage;
    GaussianBlur(image, filteredImage, Size(apertureSize, apertureSize), apertureSize / 6.0);

    // Compute gradient.
    Mat gradX, gradY;
    Sobel(image, gradX, CV_32F, 1, 0);
    Sobel(image, gradY, CV_32F, 0, 1);

    // Find the gradient strength and direction.
    Mat strength(gradX.size(), CV_32F), angles(gradX.size(), CV_32F);
    for (int i = 0; i < gradX.rows; ++i) {
        for (int j = 0; j < gradX.cols; ++j) {
            float x = gradX.at<float>(i, j), y = gradY.at<float>(i, j);
            strength.at<float>(i, j) = sqrt(x * x + y * y);
            angles.at<float>(i, j) = atan2(y, x) * 180 / acos(-1);
        }
    }

    // Non-maximum suppression
    Mat nonMaxSupped = Mat(strength.rows-2, strength.cols-2, CV_32F);
    for (int i=1; i< strength.rows - 1; i++) {
        for (int j=1; j<strength.cols - 1; j++) {
            float Tangent = angles.at<float>(i,j);

            nonMaxSupped.at<float>(i-1, j-1) = strength.at<float>(i,j);
            //Horizontal Edge
            if (((-22.5 < Tangent) && (Tangent <= 22.5)) || ((157.5 < Tangent) && (Tangent <= -157.5)))
            {
                if ((strength.at<float>(i,j) < strength.at<float>(i,j+1)) || (strength.at<float>(i,j) < strength.at<float>(i,j-1)))
                    nonMaxSupped.at<float>(i-1, j-1) = 0;
            }
            //Vertical Edge
            if (((-112.5 < Tangent) && (Tangent <= -67.5)) || ((67.5 < Tangent) && (Tangent <= 112.5)))
            {
                if ((strength.at<float>(i,j) < strength.at<float>(i+1,j)) || (strength.at<float>(i,j) < strength.at<float>(i-1,j)))
                    nonMaxSupped.at<float>(i-1, j-1) = 0;
            }
            
            //-45 Degree Edge
            if (((-67.5 < Tangent) && (Tangent <= -22.5)) || ((112.5 < Tangent) && (Tangent <= 157.5)))
            {
                if ((strength.at<float>(i,j) < strength.at<float>(i-1,j+1)) || (strength.at<float>(i,j) < strength.at<float>(i+1,j-1)))
                    nonMaxSupped.at<float>(i-1, j-1) = 0;
            }
            
            //45 Degree Edge
            if (((-157.5 < Tangent) && (Tangent <= -112.5)) || ((22.5 < Tangent) && (Tangent <= 67.5)))
            {
                if ((strength.at<float>(i,j) < strength.at<float>(i+1,j+1)) || (strength.at<float>(i,j) < strength.at<float>(i-1,j-1)))
                    nonMaxSupped.at<float>(i-1, j-1) = 0;
            }
        }
    }

    Mat EdgeMat = Mat(nonMaxSupped.rows, nonMaxSupped.cols, nonMaxSupped.type());
    
    for (int i=0; i<nonMaxSupped.rows; i++) {
        for (int j = 0; j<nonMaxSupped.cols; j++) {
            EdgeMat.at<float>(i,j) = nonMaxSupped.at<float>(i,j);
            if(EdgeMat.at<float>(i,j) > maxVal)
                EdgeMat.at<float>(i,j) = 255;
            else if(EdgeMat.at<float>(i,j) < minVal)
                EdgeMat.at<float>(i,j) = 0;
            else {
                bool anymaxVal = false;
                bool anyBetween = false;
                for (int x=i-1; x < i+2; x++) {
                    for (int y = j-1; y<j+2; y++) {
                        if(x <= 0 || y <= 0 || EdgeMat.rows || y > EdgeMat.cols) //Out of bounds
                            continue;
                        else {
                            if(EdgeMat.at<float>(x,y) > maxVal)
                            {
                                EdgeMat.at<float>(i,j) = 255;
                                anymaxVal = true;
                                break;
                            }
                            else if(EdgeMat.at<float>(x,y) <= maxVal && EdgeMat.at<float>(x,y) >= minVal)
                                anyBetween = true;
                        }
                    }
                    if(anymaxVal)
                        break;
                }

                if(!anymaxVal && anyBetween)
                    for (int x=i-2; x < i+3; x++) {
                        for (int y = j-1; y<j+3; y++) {
                            if(x < 0 || y < 0 || x > EdgeMat.rows || y > EdgeMat.cols) //Out of bounds
                                continue;
                            else {
                                if(EdgeMat.at<float>(x,y) > maxVal){
                                    EdgeMat.at<float>(i,j) = 255;
                                    anymaxVal = true;
                                    break;
                                }
                            }
                        }
                        if(anymaxVal)
                            break;
                    }
                    if(!anymaxVal)
                        EdgeMat.at<float>(i,j) = 0;
                }
            }
    }   

    return EdgeMat;
}

#endif // CANNY_H