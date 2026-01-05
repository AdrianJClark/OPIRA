#include "OPIRALibraryGPU.h"

using namespace OPIRALibrary;
using namespace std; 

RegistrationAlgorithmGPU::RegistrationAlgorithmGPU() {
}

RegistrationAlgorithmGPU::~RegistrationAlgorithmGPU() {
}

RegistrationAlgorithmGPU::RegisteredImageGPU RegistrationAlgorithmGPU::performRegistrationGPU(IplImage* image) {
	RegisteredImageGPU fp; return fp;
}

string RegistrationAlgorithmGPU::getName() {
	return "Unknown GPU Point Based Registration Algorithm";
}

RegistrationAlgorithmGPU* RegistrationAlgorithmGPU::clone() {
	RegistrationAlgorithmGPU *r = new RegistrationAlgorithmGPU(*this);
	return r;
}

RegistrationAlgorithmGPU::RegisteredImageGPU RegistrationAlgorithmGPU::cloneTrainedMarker(RegisteredImageGPU trainedMarker) {
	//TODO	
	RegisteredImageGPU tm;
	tm.descriptors.insert(tm.descriptors.begin(), trainedMarker.descriptors.begin(), trainedMarker.descriptors.end());
	tm.keypoints.insert(tm.keypoints.begin(), trainedMarker.keypoints.begin(), trainedMarker.keypoints.end());
	
	tm.gpu_descriptors = trainedMarker.gpu_descriptors.clone();
	tm.gpu_keypoints = trainedMarker.gpu_keypoints.clone();

	
	tm.markerFilename = trainedMarker.markerFilename;

	tm.matcher = cv::gpu::BruteForceMatcher_GPU<cv::L2<float>>();
	
	return tm;
}

vector<RegistrationAlgorithmGPU::RegisteredImageGPU> RegistrationAlgorithmGPU::cloneTrainedMarker(vector<RegisteredImageGPU> trainedMarkers) {
	vector<RegisteredImageGPU> retVect;
	for (unsigned int i=0; i< trainedMarkers.size(); i++) {
		retVect.push_back(cloneTrainedMarker(trainedMarkers.at(i)));
	}
	return retVect;
}

bool RegistrationAlgorithmGPU::addMarker(Marker marker) {
	RegisteredImageGPU markerFeatures = performRegistrationGPU(marker.image);

	if (markerFeatures.keypoints.size() == 0) return false;

	markerFeatures.markerFilename = marker.name;
	trainedMarkers.push_back(markerFeatures);

	//TODO: Multi marker handling

	return true;
}

bool RegistrationAlgorithmGPU::removeMarker(Marker marker) {
	//TODO
	return false;

}

vector<PointMatches> RegistrationAlgorithmGPU::findAllMatches(IplImage* image) {
	if (trainedMarkers.size() <1) return vector<PointMatches>();

	//TODO: Proper Multi-Marker Handling
	vector<PointMatches> matches;
	for (int i=0; i<trainedMarkers.size(); i++) matches.push_back(findMatches(image, i));
	return matches;
}

PointMatches RegistrationAlgorithmGPU::findMatches(IplImage* image, int index) {
	RegisteredImageGPU imageFeatures = performRegistrationGPU(image);

	cv::gpu::GpuMat trainIdx, distance, allDist;
	trainedMarkers.at(index).matcher.knnMatch(imageFeatures.gpu_descriptors, trainedMarkers.at(index).gpu_descriptors, trainIdx, distance, allDist, 2);
	
	vector<vector<cv::DMatch>> matches;
	trainedMarkers.at(index).matcher.knnMatchDownload(trainIdx, distance, matches);

	PointMatches pMatch; pMatch.resize(matches.size());

	float thresh = 1.3;

	int c=0;
	for (int i=0; i<matches.size(); i++) {
		if (matches.at(i).at(0).distance*thresh	<= matches.at(i).at(1).distance) {
			//Get the frame point
			CvPoint fp = imageFeatures.keypoints.at(matches.at(i).at(0).queryIdx).pt;
			pMatch.featImage[c].x = fp.x; pMatch.featImage[c].y = fp.y; 

			//Get the marker point
			CvPoint mp = trainedMarkers.at(index).keypoints.at(matches.at(i).at(0).trainIdx).pt;
			pMatch.featMarker[c].x = mp.x; pMatch.featMarker[c].y = mp.y;
			c++;
		}
	}

	pMatch.count = c;

	return pMatch;
}