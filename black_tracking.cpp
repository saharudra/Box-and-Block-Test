#include<stdio.h>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int, char **) {
	VideoCapture cap("/home/tr/Downloads/15_Dec_349pm_DrYang.mp4");
	Rect bounding_rect;
	

	// Kalman Filter Initialization
    	int stateSize = 6;
    	int measSize = 4;
    	int contrSize = 0;
	unsigned int type = CV_32F;

	cv::KalmanFilter kf(stateSize, measSize, contrSize, type);
	cv::Mat state(stateSize, 1, type);
	cv::Mat meas(measSize, 1, type);
	cv::setIdentity(kf.transitionMatrix);
	kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
	kf.measurementMatrix.at<float>(0) = 1.0f;
	kf.measurementMatrix.at<float>(7) = 1.0f;
	kf.measurementMatrix.at<float>(16) = 1.0f;
	kf.measurementMatrix.at<float>(23) = 1.0f;

	kf.processNoiseCov.at<float>(0) = 1e-2;
	kf.processNoiseCov.at<float>(7) = 1e-2;
	kf.processNoiseCov.at<float>(14) = 5.0f;
	kf.processNoiseCov.at<float>(21) = 5.0f;
	kf.processNoiseCov.at<float>(28) = 1e-2;
	kf.processNoiseCov.at<float>(35) = 1e-2;

	cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-1));

	// Kalman Filter Initialization End
	
	double ticks = 0; 
	bool found = false; 

	// Check if video input is given:	
	if(!cap.isOpened()) {
		return -1;
	}

	while(true) {

		// Set up Kalman Filter:
		double preTick = ticks;
		ticks = (double) cv::getTickCount();
	
		double dT = (ticks - preTick)/cv::getTickFrequency();

		Mat frame;
		cap >> frame;

		// Kalman Tracking:
		if(found) {
			kf.transitionMatrix.at<float>(2) = dT;
			kf.transitionMatrix.at<float>(9) = dT;
			cout << "dT:" << dT << endl;
			state = kf.predict();
			cout << "State:" << state << endl;

			cv::Rect predRect;
			predRect.width = state.at<float>(4);
			predRect.height = state.at<float>(5);

			predRect.x = state.at<float>(0) - predRect.width / 2;
			predRect.y = state.at<float>(1) - predRect.height / 2;
			
			cv::Point center;
			center.x = state.at<float>(0);
			center.y = state.at<float>(1);
			cv::circle(frame, center, 2, CV_RGB(255,255,0), -1);
			cv::rectangle(frame, predRect, CV_RGB(255,255,0), 2);

			
		}
		
		
		// Color Tracking: 
		Mat blur, imgThreshold;
		cv::blur(frame,blur,Size(3,3));
        	
		cv::inRange(blur, cv::Scalar(0, 0, 0, 0), cv::Scalar(180, 255, 30, 0), imgThreshold);	
		
		

		cv::Mat kernel_square = cv::getStructuringElement(MORPH_RECT,Size(3,3),Point(-1,-1));
		cv::erode(imgThreshold,imgThreshold,kernel_square);
		cv::erode(imgThreshold,imgThreshold,kernel_square);
		imshow("IMAGE Input -  --",imgThreshold);


		vector<vector<Point> > contours;
    		vector<Vec4i> hierarchy;   
		cv::findContours(imgThreshold,contours, hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
		Scalar color( 255,255,255);
		int largest_area=0;
		int largest_contour_index=0;
		
		/*
		for( int i = 0; i< contours.size(); i++ ) // iterate through each contour. 
      		{
       			double a=contourArea( contours[i],false);  //  Find the area of contour
			
			bounding_rect=boundingRect(contours[i]);
			drawContours( frame, contours,i, color, CV_FILLED, 8, hierarchy ); // Draw the largest contour using previously stored index.
 			rectangle(frame, bounding_rect,  Scalar(0,255,0),1, 8,0); 
		 
		}
		*/

		for( int i = 0; i< contours.size(); i++ ) // iterate through each contour. 
      			{
       				double a=contourArea( contours[i],false);  //  Find the area of contour
     	  			if(a>largest_area ){
       					largest_area=a;
       					largest_contour_index=i; //Store the index of largest contour
       					bounding_rect=boundingRect(contours[largest_contour_index]); // Find the bounding rectangle for biggest contour
       				}
  
      			}
 		
		/// Get the moments
  		vector<Moments> mu(contours.size() );
  		for( int i = 0; i < contours.size(); i++ )
     			{ mu[i] = moments( contours[i], false ); }
		
		///  Get the mass centers:
  		vector<Point2f> mc( contours.size() );
  		for( int i = 0; i < contours.size(); i++ )
     		{ mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }
		
		//Point<Point2f> mc_l = mc[largest_contour_index];
		cout << "Largest index of contour " << largest_contour_index << endl;
		cout << " Mass center of the largest contour " << mc[largest_contour_index] << endl;


		drawContours( frame, contours,largest_contour_index, color, CV_FILLED, 8, hierarchy ); // Draw the largest contour using previously stored index.
 		rectangle(frame, bounding_rect,  Scalar(0,255,0),1, 8,0); 
		

		imshow("Image",imgThreshold);
		imshow("IMAGE Input", frame);
		
		waitKey(100);
	}
}
	
