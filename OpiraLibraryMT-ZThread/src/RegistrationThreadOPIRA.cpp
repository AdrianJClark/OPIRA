#include "OPIRALibraryMT-ZThread.h"
#include "highgui.h"

using namespace OPIRALibrary;

RegistrationThreadOPIRA::RegistrationThreadOPIRA(RegistrationAlgorithm *registrationAlgorithm, int markerIndex, CvSize markerSize):RegistrationThread(registrationAlgorithm) {
	mIndex = markerIndex; mSize = markerSize; newHomography = 0;
}

RegistrationThreadOPIRA::~RegistrationThreadOPIRA() {
	if (curFrame) cvReleaseImage(&curFrame); if (newFrame) cvReleaseImage(&newFrame);
	if (newHomography) cvReleaseMat(&newHomography);
	curMatches.clear();
}


void RegistrationThreadOPIRA::run() {
	while (running) {
		if (updated) {
			newMutex.acquire();	
				IplImage *unDistort = cvCreateImage(mSize, newFrame->depth, newFrame->nChannels);
				cvWarpPerspective(newFrame, unDistort, newHomography, 1+8+16);
				PointMatches newMatches =  regAlgorithm->findMatches(unDistort, mIndex);
				cvReleaseImage(&unDistort);
				if (newMatches.count>0) {
					CvMat mMatches = cvMat(newMatches.count, 1, CV_32FC2, newMatches.featImage);
					cvPerspectiveTransform(&mMatches, &mMatches, newHomography);
				}
			newMutex.release();


			curMutex.acquire(); 
				curMatches.clear();
				curMatches = Ransac(newMatches);
				newMutex.acquire(); 
					cvReleaseImage(&curFrame); curFrame = cvCloneImage(newFrame); 
					cvReleaseImage(&newFrame); newFrame = 0;
					cvReleaseMat(&newHomography); newHomography = 0;
				newMutex.release();
			curMutex.release();

			newMatches.clear(); updated = false;
		} else {
			Thread::sleep(1);
		}
	}
};

PointMatches RegistrationThreadOPIRA::undistortRegister(IplImage *frame_input, CvMat *homography, CvSize frameSize) {
	PointMatches returnMatches; 

	if (!updated) {
		newMutex.acquire(); 
			newFrame = cvCloneImage(frame_input); newHomography = cvCloneMat(homography); updated = true;
			mSize = frameSize;
		newMutex.release(); 
	}

	if (curFrame) {
		//Copy the last known good registration
		curMutex.acquire(); 
		PointMatches m; m.clone(curMatches);
		IplImage *f = cvCloneImage(curFrame);
		curMutex.release(); 

		//Calculate the latest position for all the matches
		returnMatches.clear();
		returnMatches = opticalFlow(f, m, frame_input);
		m.clear();

		//Clean up
		cvReleaseImage(&f); m.clear();
	}

	return returnMatches;
}