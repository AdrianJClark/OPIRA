#include "OPIRALibrary.h"
#include "opencv/highgui.h"

using namespace OPIRALibrary;
using namespace std; 

RegistrationOpticalFlow::RegistrationOpticalFlow(RegistrationAlgorithm *registrationAlgorithm): RegistrationStandard(registrationAlgorithm) 
	{
		previousImage = 0; previousPyramid = 0;
	}

bool RegistrationOpticalFlow::addMarker(string markerFilename) {
	if (RegistrationStandard::addMarker(markerFilename)) {
		previousMatches.push_back(PointMatches());
		return true;
	}
	return false;
}

bool RegistrationOpticalFlow::addResizedMarker(string markerName, int maxLengthSize) {
	if (RegistrationStandard::addResizedMarker(markerName, maxLengthSize)) {
		previousMatches.push_back(PointMatches());
		return true;
	}
	return false;
}

bool RegistrationOpticalFlow::addScaledMarker(string markerName, int maxLengthScale) {
	if (RegistrationStandard::addScaledMarker(markerName, maxLengthScale)) {
		previousMatches.push_back(PointMatches());
		return true;
	}
	return false;
}

bool RegistrationOpticalFlow::addResizedScaledMarker(string markerName, int maxLengthSize, int maxLengthScale) {
	if (RegistrationStandard::addResizedScaledMarker(markerName, maxLengthSize, maxLengthScale)) {
		previousMatches.push_back(PointMatches());
		return true;
	}
	return false;
}

bool RegistrationOpticalFlow::removeMarker(string markerFilename) {
	for (unsigned int i=0; i<markers.size(); i++) {
		if (markers.at(i).name == markerFilename) {
			previousMatches.at(i).clear();
			previousMatches.erase(previousMatches.begin()+i); 
			RegistrationStandard::removeMarker(markerFilename);
			return true;
		}
	}
	return false;
}

RegistrationOpticalFlow::~RegistrationOpticalFlow() {
	for (unsigned int i=0; i< previousMatches.size(); i++) previousMatches.at(i).clear();
	if (previousImage) cvReleaseImage(&previousImage);
	if (previousPyramid) cvReleaseImage(&previousPyramid);
}

vector<MarkerTransform> RegistrationOpticalFlow::performRegistration(IplImage* frame_input, CvMat* captureParams, CvMat* captureDistortion) 
	{
	vector<MarkerTransform> retVal;
	string windowText;

	vector<PointMatches> matches = regAlgorithm->findAllMatches(frame_input);

	currentImage = cvCreateImage(cvGetSize(frame_input), IPL_DEPTH_8U, 1); cvConvertImage(frame_input, currentImage);
	currentPyramid = cvCreateImage(cvGetSize(currentImage), IPL_DEPTH_32F, 1);

	for (unsigned int i=0; i<matches.size(); i++) {
		PointMatches *bestMatch=0;
		PointMatches pMatch = Ransac(matches.at(i));
		if (pMatch.count >=minRegistrationCount) { bestMatch = &pMatch; windowText = regAlgorithm->getName(); }
		
		//If we are able to match at least 2 points through optical flow
		PointMatches optFlow = opticalFlow(previousMatches.at(i));
		if (optFlow.count >=minOptFlowCount && optFlow.count>=pMatch.count) { bestMatch = &optFlow; windowText = "Optical Flow"; }
		previousMatches.at(i).clear();

		if (bestMatch !=0) {
			//Copy the points into the previous array
			previousMatches.at(i).clone(*bestMatch);

			//Display the matches if we feel so inclined
			if (displayImage) displayMatches(markers.at(i).image, frame_input, *bestMatch, regAlgorithm->getName() + " Optical Flow", windowText, 1.0);

			//Calculate the OGL viewpoint and projection and transformation matrices
			retVal.push_back(computeMarkerTransform(*bestMatch, i, cvGetSize(frame_input), captureParams, captureDistortion));
		}

		//Cleanup
		matches.at(i).clear();
		pMatch.clear();
		optFlow.clear();
	}

	cvReleaseImage(&previousImage); previousImage = currentImage; currentImage = 0;
	cvReleaseImage(&previousPyramid); previousPyramid = currentPyramid; currentPyramid = 0;

	return retVal;
}


/*	int getMatchCount() {
		return previousMatches.count;
	}
*/

PointMatches RegistrationOpticalFlow::opticalFlow(PointMatches prevMatches) {
	if (prevMatches.count==0 || previousImage ==0) return PointMatches();
	if (previousImage->width != currentImage->width || previousImage->height != currentImage->height)  return PointMatches();

	CvPoint2D32f *currentFeatures = (CvPoint2D32f*)cvAlloc(prevMatches.count*sizeof(CvPoint2D32f));
	PointMatches flowMatches;

	char* status = (char *)malloc(prevMatches.count*sizeof(char));
	
	cvCalcOpticalFlowPyrLK(previousImage, currentImage, previousPyramid, currentPyramid, (CvPoint2D32f *)prevMatches.featImage, currentFeatures, prevMatches.count, cvSize(10,10), 3, status, 0, cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03), CV_LKFLOW_PYR_A_READY);
	//cvCalcOpticalFlowPyrLK(grayPrev_image, grayCurrent_image, 0, 0, (CvPoint2D32f *)prevMatches.featImage, currentFeatures, prevMatches.count, cvSize(10,10), 3, status, 0, cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03), 0 );


	flowMatches.resize(prevMatches.count); flowMatches.count =0;
	for (int i=0; i<prevMatches.count; i++) {
		if(status[i]==1) {
			flowMatches.featMarker[flowMatches.count].x = (float)cvRound(prevMatches.featMarker[i].x); flowMatches.featMarker[flowMatches.count].y = (float)cvRound(prevMatches.featMarker[i].y);
			flowMatches.featImage[flowMatches.count].x = (float)cvRound(currentFeatures[i].x); flowMatches.featImage[flowMatches.count].y = (float)cvRound(currentFeatures[i].y);
			flowMatches.count++;
		}
	}

	PointMatches returnMatches = Ransac(flowMatches);
	flowMatches.clear();

	cvFree((void**)&currentFeatures);
	free(status);
	
	return returnMatches;
}