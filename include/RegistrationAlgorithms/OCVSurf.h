#ifndef POINTBASEDOCVSURFALGORITHM_H
#define POINTBASEDOCVSURFALGORITHM_H

#include "OPIRALibrary.h"
using namespace OPIRALibrary;

class OCVSurf: public RegistrationAlgorithm {
public:
	OCVSurf(bool descriptor128bit=false, int nOctaves=4, int nOctaveLayers=3,  double blobThresh=400, int nSamplingSteps=2); 
	~OCVSurf();
	RegistrationAlgorithm* clone();

	std::string getName();
	FeatureVector performRegistration(IplImage* image);

	

private:
	CvSeq* fastHessianDetector( const CvMat* sum, const CvMat* mask_sum, CvMemStorage* storage, const CvSURFParams* params );

	cv::SURF surf;
	int extended; // 0 means basic descriptors (64 elements each),
                  // 1 means extended descriptors (128 elements each)
    double hessianThreshold; // only features with keypoint.hessian larger than that are extracted.
                  // good default value is ~300-500 (can depend on the average
                  // local contrast and sharpness of the image).
                  // user can further filter out some features based on their hessian values
                  // and other characteristics
    int octaves; // the number of octaves to be used for extraction.
                  // With each next octave the feature size is doubled (3 by default)
    int octaveLayers; // The number of layers within each octave (4 by default)
	int samplingSteps;		// Sampling step along image x and y axes at first octave. This is doubled
							// for each additional octave. WARNING: Increasing this improves speed, 
							// however keypoint extraction becomes unreliable.

};

#endif