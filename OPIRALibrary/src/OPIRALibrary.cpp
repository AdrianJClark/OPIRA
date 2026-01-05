#include "OPIRALibrary.h"
#include "opencv/highgui.h"

using namespace std; 

void OPIRALibrary::displayMatches(IplImage *marker, IplImage *scene, PointMatches matches, string windowName, string text,  float markerScale, float frameScale) {
	//Resize the marker
	IplImage* markerResize = cvCreateImage(cvSize((int)((float)marker->width*markerScale), (int)((float)marker->height*markerScale)), marker->depth, marker->nChannels);
	cvResize(marker, markerResize);

	//Resize the scene
	IplImage* frameResize = cvCreateImage(cvSize((int)((float)scene->width*frameScale), (int)((float)scene->height*frameScale)), scene->depth, scene->nChannels);
	cvResize(scene, frameResize);

	//Create Image	
	int imWidth = markerResize->width + frameResize->width;
	int imHeight = markerResize->height>frameResize->height ? markerResize->height: frameResize->height;
	IplImage *image = cvCreateImage(cvSize(imWidth, imHeight), IPL_DEPTH_8U, 3);

	if (frameResize->nChannels==1) {
		IplImage *frameResizeCol = cvCreateImage(cvGetSize(frameResize), IPL_DEPTH_8U, 3);
		cvMerge(frameResize, frameResize, frameResize, 0, frameResizeCol);
		//Copy Scene into Image
		cvSetImageROI(image, cvRect(0,0,frameResizeCol->width, frameResizeCol->height));
		cvRepeat(frameResizeCol, image);
		cvReleaseImage(&frameResizeCol);
	} else {
		//Copy Scene into Image
		cvSetImageROI(image, cvRect(0,0,frameResize->width, frameResize->height));
		cvRepeat(frameResize, image);
	}

	if (markerResize->nChannels==1) {
		IplImage *markerResizeCol = cvCreateImage(cvGetSize(markerResize), IPL_DEPTH_8U, 1);
		cvMerge(markerResize, markerResize, markerResize, 0, markerResizeCol);
		//Copy Marker into Image
		cvSetImageROI(image, cvRect(frameResize->width, 0, markerResizeCol->width, markerResizeCol->height));
		cvRepeat(markerResizeCol, image);
		cvReleaseImage(&markerResizeCol);
	} else {
		cvSetImageROI(image, cvRect(frameResize->width, 0, markerResize->width, markerResize->height));
		cvRepeat(markerResize, image);
	}
	cvResetImageROI(image);

	//Draw Matches
	for(int i=0; i<matches.count; i++) {
		CvPoint2D32f imgFeat = matches.featImage[i]; imgFeat.x*=frameScale; imgFeat.y*=frameScale;
		CvPoint2D32f markFeat = matches.featMarker[i]; markFeat.y*=markerScale; markFeat.x = (markFeat.x*markerScale)+frameResize->width;
			cvCircle(image, cvPoint((int)imgFeat.x, (int)imgFeat.y), 3, cvScalar(0,255,255));
			cvCircle(image, cvPoint((int)markFeat.x, (int)markFeat.y), 3, cvScalar(0,255,255));
			cvLine(image, cvPoint((int)imgFeat.x, (int)imgFeat.y),cvPoint((int)markFeat.x, (int)markFeat.y), cvScalar(0,255,255));
	}

	if (text.length()>0) {
		CvFont font;
		cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 8);
		cvPutText(image, text.c_str(), cvPoint(5,20), &font, cvScalar(255,255,255,0));
	}

	//Display and Clean up
	cvShowImage(windowName.c_str(), image);
	cvReleaseImage(&image);
	cvReleaseImage(&markerResize);
	cvReleaseImage(&frameResize);
}