#ifndef POINTBASEDSURFALGORITHM_H
#define POINTBASEDSURFALGORITHM_H

#include "OPIRALibrary.h"
using namespace OPIRALibrary;

class SURF: public RegistrationAlgorithm {
public:
	SURF(bool rotationInvariant=false, bool descriptor128bit=false, bool doubleImageSize=false, int nOctaves=4, int nSamplingSteps = 2, int initalLobeSize=3, double blobThres=4.0, int windowSize = 4);
	~SURF();
	RegistrationAlgorithm* clone();

	string getName();
	FeatureVector performRegistration(IplImage* image);


private:
	// Blob response treshold
	double thres;
	// Initial lobe size, default 3 and 5 (with double image size)
	int initLobe;
	// Initial sampling step (default 2)
	int samplingStep;
	// Number of analysed octaves (default 4)
	int octaves;
    // Set this flag "true" to double the image size
    bool doubleImageSize;

    // Upright SURF or rotation invaraiant
    bool upright;
	// If the extended flag is turned on, SURF 128 is used
	bool extended;
	// Spatial size of the descriptor window (default 4)
	int indexSize;
};

#endif