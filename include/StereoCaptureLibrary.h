#ifndef STEREOCAPTURELIBRARY_H
#define STEREOCAPTURELIBRARY_H

#include <cv.h>
#include <highgui.h>
#include "videoInput.h"
#include <iostream>

class CaptureStereo {
public:
	CaptureStereo() { captureParams=captureDistortion=mDistortX=mDistortY=camRectify=camProjection=camReprojection=0; captureUndistort=false;}
	virtual ~CaptureStereo() {
		if (captureParams) cvReleaseMat(&captureParams);
		if (captureDistortion) cvReleaseMat(&captureDistortion);
		if (camRectify) cvReleaseMat(&camRectify);
		if (camProjection) cvReleaseMat(&camProjection);
		if (camReprojection) cvReleaseMat(&camReprojection);
		if (mDistortX) cvReleaseMat(&mDistortX);
		if (mDistortY) cvReleaseMat(&mDistortY);
	}

	virtual IplImage* getFrame() { return 0;}
	CvMat* getParameters() { return captureParams; }
	CvMat* getDistortion() { if (captureUndistort) return 0; else return captureDistortion; }

	int getWidth() { return captureWidth; }
	int getHeight() { return captureHeight; }
	
	bool getUndistort() { return captureUndistort; }
	void setUndistort(bool undistort) { 
		captureUndistort = undistort; 
		if (captureUndistort) { 
			captureParams->data.db[2] = captureWidth/2.0; captureParams->data.db[5] = captureHeight/2.0; 
		} else {
			captureParams->data.db[2] = principalX; captureParams->data.db[5] = principalY; 
		}
	}

	/*Return the reprojection matrix*/
	CvMat* getReprojection() {
		return camReprojection;
	}

protected:
	CvMat* captureParams, *captureDistortion;
	CvMat* camProjection, *camReprojection, *camRectify;
	CvMat* mDistortX, *mDistortY;
	int captureWidth, captureHeight;
	bool captureUndistort;
	int principalX, principalY;

	bool loadCaptureParams(string camera_params) {
		//Create some storage and open the file
		CvMemStorage* storage = cvCreateMemStorage();
		CvFileStorage* fstorage = cvOpenFileStorage(camera_params.c_str(), storage, CV_STORAGE_READ);
		//If something went wrong, print an error message
		if(!fstorage)
		{
			cerr << "Failed to open calibration file " << camera_params.c_str() << endl;
			return false;
		}
		
		//Load the information out of the file
		captureWidth = cvReadIntByName(fstorage, NULL, "image_width");
		captureHeight = cvReadIntByName(fstorage, NULL, "image_height");
		captureParams = (CvMat*)cvReadByName(fstorage, NULL, "camera_matrix");
		captureDistortion = (CvMat*)cvReadByName(fstorage, NULL, "distortion_coefficients");
		camRectify = (CvMat*)cvReadByName(fstorage, NULL, "R");
		camProjection = (CvMat*)cvReadByName(fstorage, NULL, "P");
		camReprojection = (CvMat*)cvReadByName(fstorage, NULL, "Q");
		principalX = captureParams->data.db[2]; principalY = captureParams->data.db[5];

		//If any of the matrices failed to load
		if(!captureParams || !captureDistortion || !camRectify || !camProjection || !camReprojection)
		{
			// print an error message
			cerr << "Failed to read intrinsic parameters from " << camera_params.c_str() << endl;
			return false;
		} else {
			//Otherwise initialise the undistortion maps
			mDistortX = cvCreateMat(captureHeight, captureWidth, CV_32F);
			mDistortY = cvCreateMat(captureHeight, captureWidth, CV_32F);
			cvInitUndistortRectifyMap(captureParams, captureDistortion, camRectify, camProjection, mDistortX, mDistortY);
		}
	    
		//Clean up
		cvReleaseFileStorage(&fstorage);
		cvReleaseMemStorage(&storage);
		return true;
	} 
};

class CameraStereo: public CaptureStereo {
public:
	CameraStereo() {init(0, cvSize(320,240), 0);}
	CameraStereo(char* parametersFile) {init(0, cvSize(320,240), parametersFile);}
	CameraStereo(int cameraIndex) {init(cameraIndex, cvSize(320,240), 0);};
	CameraStereo(CvSize imgSize) {init(0, imgSize, 0);};
	CameraStereo(int cameraIndex, CvSize imgSize) {init(cameraIndex, imgSize, 0);};
	CameraStereo(int cameraIndex, char* parametersFile) {init(cameraIndex, cvSize(320,240), parametersFile);};
	CameraStereo(int cameraIndex, CvSize imgSize, char* parametersFile) {init(cameraIndex, imgSize, parametersFile);};
	~CameraStereo() { delete vi; }

	IplImage* getFrame()	{
		if (camIndex >= vi->devicesFound || captureWidth ==0 || captureHeight ==0 ) return 0;
		IplImage* newFrame = cvCreateImage(cvSize(captureWidth, captureHeight), IPL_DEPTH_8U, 3);
		if (vi) newFrame->imageData = (char*)vi->getPixels(camIndex, false, true);
		
		if (captureUndistort) {
			IplImage *unDistortedFrame = cvCreateImage(cvGetSize(newFrame), newFrame->depth, newFrame->nChannels);
			cvRemap( newFrame, unDistortedFrame, mDistortX, mDistortY);
			cvReleaseImage(&newFrame); newFrame = unDistortedFrame;
		}

		return newFrame;
	}

	/*Find out if auto white balancing is on*/
	bool getAutoWhiteBalance() {
		long pMin, pMax, pStep, pVal, pFlags, pDef;
		//Query the camera
		vi->getVideoSettingFilter(camIndex, vi->propWhiteBalance, pMin, pMax, pStep, pVal, pFlags, pDef); 
		//if pFlags is 1, auto white balancing is on
		return (pFlags==1);
	}

	/*Turn white balancing on or off*/
	void setAutoWhiteBalance(bool whiteBal, int value=-1) {
		//Get the current value
		long pMin, pMax, pStep, pVal, pFlags, pDef;
		vi->getVideoSettingFilter(camIndex, vi->propWhiteBalance, pMin, pMax, pStep, pVal, pFlags, pDef); 

		if (value>=pMin && value<=pMax) pVal = value;
		//The value is 1 if white balancing is on, 2 if it isn't
		if (whiteBal)
			vi->setVideoSettingFilter(camIndex, vi->propWhiteBalance, pVal, 1); 
		else
			vi->setVideoSettingFilter(camIndex, vi->propWhiteBalance, pVal, 2); 
		
	}

protected:
	void init(int cameraIndex, CvSize imgSize, char* parametersFile) {
		camIndex = cameraIndex;
		vi = new videoInput();
		vi->setVerbose(false); vi->setUseCallback(true);

		if (parametersFile && strlen(parametersFile)>0) { 
			loadCaptureParams(parametersFile); 
			setUndistort(true); 
			if (captureWidth != imgSize.width || captureHeight !=imgSize.height) {
				std::cerr << "Calibration File Resolution does not match camera resolution, changing camera resolution" << std::endl;
				imgSize.width = captureWidth; imgSize.height = captureHeight;
			}
		}

		vi->setupDevice(cameraIndex, imgSize.width, imgSize.height);
		captureWidth = vi->getWidth(cameraIndex); captureHeight = vi->getHeight(cameraIndex);

		if (vi->devicesFound==0) std::cerr << "No Cameras Found" << std::endl;
		else if (cameraIndex >= vi->devicesFound) std::cerr << "Invalid Camera Index" << std::endl;
		else if (captureWidth == 0 || captureHeight == 0) std::cerr << "Invalid Camera Resolution" << std::endl;
	}

	videoInput *vi;
	int camIndex;
};

class VideoStereo: public CaptureStereo {
public:
	VideoStereo(char *videoFile) { init(videoFile, 0); }
	VideoStereo(char *videoFile, char* parametersFile) { init(videoFile, parametersFile); }

	~VideoStereo() { if (captureObj) cvReleaseCapture( &captureObj ); }
	
	IplImage* getFrame() { 	
		IplImage *newFrame;

		//Make sure we have a capture object
		if (captureObj) {
			if( !cvGrabFrame( captureObj )) return 0;
		}
		//Make a copy of the image
		newFrame = cvCloneImage(cvRetrieveFrame( captureObj ));

		if (captureUndistort && captureParams!=0) {
			//If we are undistorting, create a new Frame, undistort into it, then release the original
			//Frame and set it's pointer to the undistortedFrame
			IplImage *unDistortedFrame = cvCreateImage(cvGetSize(newFrame), newFrame->depth, newFrame->nChannels);
			cvRemap( newFrame, unDistortedFrame, mDistortX, mDistortY);
			//cvUndistort2( newFrame, unDistortedFrame, captureParams, captureDistortion );
			cvReleaseImage(&newFrame); newFrame = unDistortedFrame;
		}

		if( newFrame->origin == IPL_ORIGIN_BL ) {
			//If the frame is upside down, flip it the right way up
			IplImage *flippedFrame = cvCreateImage( cvGetSize(newFrame), newFrame->depth, newFrame->nChannels);
			cvFlip( newFrame, flippedFrame, 0 );
			cvReleaseImage(&newFrame); newFrame = flippedFrame;
		}

		//Return the new frame
		return newFrame;
	}

private:
	CvCapture *captureObj;

	void init(char *videoFile, char *parametersFile) {
		captureObj = cvCreateFileCapture(videoFile);
		if (!captureObj) { 
			std::cerr << "Unable to load video file: " << videoFile << std::endl; 
			exit(0); 
		}

		captureUndistort=false;
		if (parametersFile!=0) {
			if (!loadCaptureParams(parametersFile)) { 
				std::cerr << "Unable to camera parameter file: " << parametersFile << std::endl; 
				cvReleaseCapture(&captureObj); 
				exit(0); 
			}
			setUndistort(true);
		}

		if (captureWidth==-1) captureWidth = (int)cvGetCaptureProperty(captureObj, CV_CAP_PROP_FRAME_WIDTH); 
		if (captureHeight==-1) captureHeight = (int)cvGetCaptureProperty(captureObj, CV_CAP_PROP_FRAME_HEIGHT);	
		
	}
};

#endif
