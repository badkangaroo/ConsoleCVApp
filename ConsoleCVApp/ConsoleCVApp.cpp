// ConsoleCVApp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;
int main()
{
	Mat img = imread("image.jpg", CV_LOAD_IMAGE_ANYCOLOR);
	namedWindow("window", WINDOW_OPENGL);
	cout << img.cols;
	// the Mat data type
	Mat gimg = img.clone();
	
	imshow("window", gimg);
	waitKey(0);
	return 0;
}

