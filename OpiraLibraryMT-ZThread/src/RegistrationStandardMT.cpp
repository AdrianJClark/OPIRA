#include "OPIRALibraryMT-ZThread.h"
#include "highgui.h"
#include <iostream>

using namespace OPIRALibrary;
using namespace ZThread;

RegistrationStandardMT::RegistrationStandardMT(RegistrationAlgorithm *registrationAlgorithm):RegistrationStandard(registrationAlgorithm) 
{
	regThread = new RegistrationThread(registrationAlgorithm->clone()); thread = new Thread(regThread);
}

RegistrationStandardMT::~RegistrationStandardMT() 
{
	regThread->stop();
	while (!regThread->isFinished()) thread->yield();
	delete thread;
}

bool RegistrationStandardMT::removeMarker(string markerName) {
	for (unsigned int i=0; i<markers.size(); i++) {
		if (markers.at(i).name == markerName) {
			//Remove it from the registration algorithm and clean up
			regThread->removeMarker(markers.at(i));
			cvReleaseImage(&markers.at(i).image);
			markers.erase(markers.begin()+i);
			return true;
		}
	}
	return false;
}

bool RegistrationStandardMT::addMarker(string markerName) {
	//Create a new marker object and set it up
	Marker marker; marker.image = cvLoadImage(markerName.c_str());

	if (marker.image==0) { std::cerr << "Cannot Load marker: " << markerName << endl; return false; }
	marker.size = cvGetSize(marker.image);
	marker.name = markerName;

	//Attempt to register it, if it succeeds, add it to the list of trained markers
	if (regThread->addMarker(marker)) {
		markers.push_back(marker);
	} else {
		std::cerr << "Unable to find any interest points in marker: " << markerName << endl;  return false;
	}

	return true;
}

bool RegistrationStandardMT::addResizedMarker(string markerName, int maxLengthSize) {
	//Create a new marker object and set it up
	Marker marker; IplImage *origImage	= cvLoadImage(markerName.c_str());
	if (origImage==0) { std::cerr << "Cannot Load marker: " << markerName << endl; return false; }

	//If the marker isn't already scaled
	if (max(origImage->width, origImage->height) != maxLengthSize) {
		//Calculate the new size
		if (origImage->width > origImage->height) {
			marker.size = cvSize(maxLengthSize, int(maxLengthSize*float(origImage->height)/float(origImage->width)));
		} else {
			marker.size = cvSize(int(maxLengthSize*float(origImage->width)/float(origImage->height)), maxLengthSize);
		}
		//Resize the image
		marker.image = cvCreateImage(marker.size, origImage->depth, origImage->nChannels);
		cvResize(origImage, marker.image);
		cvReleaseImage(&origImage);
	} else {
		//If it is scaled, just copy the reference
		marker.image = origImage;
	}

	marker.size = cvGetSize(marker.image);
	marker.name = markerName;

	//Attempt to register it, if it succeeds, add it to the list of trained markers
	if (regThread->addMarker(marker)) {
		markers.push_back(marker);
	} else {
		std::cerr << "Unable to find any interest points in marker: " << markerName << endl;  return false;
	}

	return true;
}

bool RegistrationStandardMT::addScaledMarker(string markerName, int maxLengthScale) {
	//Create a new marker object and set it up
	Marker marker; marker.image = cvLoadImage(markerName.c_str());
	if (marker.image==0) { std::cerr << "Cannot Load marker: " << markerName << endl; return false; }
	marker.name = markerName;

	//If the marker isn't already scaled
	if (max(marker.image->width, marker.image->height) != maxLengthScale) {
		//Calculate the new size
		if (marker.image->width > marker.image->height) {
			marker.size = cvSize(maxLengthScale, int(maxLengthScale*float(marker.image->height)/float(marker.image->width)));
		} else {
			marker.size = cvSize(int(maxLengthScale*float(marker.image->width)/float(marker.image->height)), maxLengthScale);
		}
	} else {
		//If it is scaled, just copy the reference
		marker.size = cvGetSize(marker.image);
	}

	//Attempt to register it, if it succeeds, add it to the list of trained markers
	if (regThread->addMarker(marker)) {
		markers.push_back(marker);
	} else {
		std::cerr << "Unable to find any interest points in marker: " << markerName << endl;  return false;
	}

	return true;
}

bool RegistrationStandardMT::addResizedScaledMarker(string markerName, int maxLengthSize, int maxLengthScale) {
	//Create a new marker object and set it up
	Marker marker; IplImage *origImage	= cvLoadImage(markerName.c_str());
	if (origImage==0) { std::cerr << "Cannot Load marker: " << markerName << endl; return false; }

	//If the marker isn't already scaled
	if (max(origImage->width, origImage->height) != maxLengthSize) {
		//Calculate the new size
		if (origImage->width > origImage->height) {
			marker.size = cvSize(maxLengthSize, int(maxLengthSize*float(origImage->height)/float(origImage->width)));
		} else {
			marker.size = cvSize(int(maxLengthSize*float(origImage->width)/float(origImage->height)), maxLengthSize);
		}
		//Resize the image
		marker.image = cvCreateImage(marker.size, origImage->depth, origImage->nChannels);
		cvResize(origImage, marker.image);
		cvReleaseImage(&origImage);
	} else {
		//If it is scaled, just copy the reference
		marker.image = origImage;
	}

	//If the marker isn't already scaled
	if (max(marker.image->width, marker.image->height) != maxLengthScale) {
		//Calculate the new size
		if (marker.image->width > marker.image->height) {
			marker.size = cvSize(maxLengthScale, int(maxLengthScale*float(marker.image->height)/float(marker.image->width)));
		} else {
			marker.size = cvSize(int(maxLengthScale*float(marker.image->width)/float(marker.image->height)), maxLengthScale);
		}
	} else {
		//If it is scaled, just copy the reference
		marker.size = cvGetSize(marker.image);
	}

	marker.name = markerName;

	//Attempt to register it, if it succeeds, add it to the list of trained markers
	if (regThread->addMarker(marker)) {
		markers.push_back(marker);
	} else {
		std::cerr << "Unable to find any interest points in marker: " << markerName << endl;  return false;
	}

	return true;
}


vector<MarkerTransform> RegistrationStandardMT::performRegistration(IplImage* frame_input, CvMat* captureParams, CvMat* captureDistortion) 
{
	vector<MarkerTransform> retVal;
	//Find all potential markers in the image
	vector<PointMatches> matches = regThread->findAllMatches(frame_input);

	//Loop through each potential match
	for (unsigned int i=0; i<matches.size(); i++) {
		//Run RANSAC to remove erroneous points
		PointMatches pMatch = Ransac(matches.at(i));

		//If we have enough matches, add it to the list of found markers
		if (pMatch.count >= minRegistrationCount && i < markers.size()) {
			if (displayImage) displayMatches(markers.at(i).image, frame_input, pMatch, regAlgorithm->getName() + " StandardMT", "", 1.0);
			retVal.push_back(computeMarkerTransform(pMatch, i, cvGetSize(frame_input), captureParams, captureDistortion));
		} 
		//Clean up
		pMatch.clear();
		matches.at(i).clear();
	}

	return retVal;
}