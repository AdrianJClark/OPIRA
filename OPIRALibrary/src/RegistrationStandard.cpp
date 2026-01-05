#include "OPIRALibrary.h"
#include "opencv/highgui.h"

#include <windows.h>

using namespace OPIRALibrary;
using namespace std; 

/* Constructor with a no markers */
RegistrationStandard::RegistrationStandard(RegistrationAlgorithm *RegistrationAlgorithm) {
	//Store the registration algorithm, and load the marker
	regAlgorithm = RegistrationAlgorithm; 
}

/* Destructor */
RegistrationStandard::~RegistrationStandard() {
	//Release the registration algorithm
	delete regAlgorithm;
}

/* Perform registration on an image */
vector<MarkerTransform> RegistrationStandard::performRegistration(IplImage* frame_input, CvMat* captureParams, CvMat* captureDistortion) 
{
	vector<MarkerTransform> retVal;
	//Find all potential markers in the image
	vector<PointMatches> matches; matches.push_back(regAlgorithm->findMatches(frame_input, 0));

	//Loop through each potential match
	for (unsigned int i=0; i<matches.size(); i++) {

		//Run RANSAC to remove erroneous points
		PointMatches pMatch = Ransac(matches.at(i));

		//If we have enough matches, add it to the list of found markers
		if (pMatch.count >= minRegistrationCount) {
			if (displayImage) displayMatches(markers.at(i).image, frame_input, pMatch, regAlgorithm->getName() + " Standard", "", 1.0);
			retVal.push_back(computeMarkerTransform(pMatch, i, cvGetSize(frame_input), captureParams, captureDistortion));
		}

		//Clean up
		pMatch.clear();
		matches.at(i).clear();
	}

	return retVal;
}

