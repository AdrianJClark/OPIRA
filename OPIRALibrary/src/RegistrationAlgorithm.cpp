#include "OPIRALibrary.h"
#include "opencv/highgui.h"

using namespace OPIRALibrary;
using namespace std; 

RegistrationAlgorithm::FeatureVector RegistrationAlgorithm::performRegistration(IplImage* image) {
	FeatureVector fp; return fp;
}

RegistrationAlgorithm::RegistrationAlgorithm() {
	allMarkers.numPoints = 0; allMarkers.features = (CvPoint2D32f*)malloc(0);
	allMarkers.descriptors = 0; allMarkers.tree = 0;
}

string RegistrationAlgorithm::getName()
{
	return "Unknown Point Based Registration Algorithm";
}

RegistrationAlgorithm::~RegistrationAlgorithm() {
	free(allMarkers.features);
	if (allMarkers.tree!=0) delete allMarkers.tree;
	if (allMarkers.descriptors!=0) delete allMarkers.descriptors;
	
	for (unsigned int i=0; i< trainedMarkers.size(); i++) {
		trainedMarkers.at(i).descriptors->release();
		free(trainedMarkers.at(i).features);
		delete trainedMarkers.at(i).tree;
		delete trainedMarkers.at(i).descriptors;
	}
}

RegistrationAlgorithm* RegistrationAlgorithm::clone() {
	RegistrationAlgorithm *r = new RegistrationAlgorithm(*this);
	return r;
}

RegistrationAlgorithm::TrainedMarker RegistrationAlgorithm::cloneTrainedMarker(TrainedMarker trainedMarker) {
		TrainedMarker tm; tm.numPoints = trainedMarker.numPoints;
		tm.descriptorLength = trainedMarker.descriptorLength;
		if (tm.numPoints>0) {
			tm.descriptors = new cv::Mat(* trainedMarker.descriptors);
			tm.tree = new cv::flann::Index(*(tm.descriptors), cv::flann::KDTreeIndexParams(4));  // using 4 randomized kdtrees
		} else {
			tm.tree = 0; tm.descriptors = 0;
		}
		tm.features = (CvPoint2D32f*)malloc(sizeof(CvPoint2D32f) * tm.numPoints); memcpy(tm.features, trainedMarker.features, tm.numPoints*sizeof(CvPoint2D32f));
		
		tm.markerFilename = trainedMarker.markerFilename;
		return tm;
}

vector<RegistrationAlgorithm::TrainedMarker> RegistrationAlgorithm::cloneTrainedMarker(vector<TrainedMarker> trainedMarkers) {
	vector<TrainedMarker> retVect;
	for (unsigned int i=0; i< trainedMarkers.size(); i++) {
		retVect.push_back(cloneTrainedMarker(trainedMarkers.at(i)));
	}
	return retVect;
}

bool RegistrationAlgorithm::addMarker(Marker marker) {
	FeatureVector markerFeatures = performRegistration(marker.image);

	if (markerFeatures.size() == 0) {
		markerFeatures.clear();
		return false;
	}

	//Create new Trained Marker
	TrainedMarker tm = trainMarker(markerFeatures); 
	tm.markerFilename = marker.name;
	trainedMarkers.push_back(tm);

	//Add to set of trained markers
	{
		//Record the size of the new marker
		if (allMarkersPos.size()>0) {
			allMarkersPos.push_back(allMarkersPos.at(allMarkersPos.size()-1)+markerFeatures.size());
		} else {
			allMarkersPos.push_back(markerFeatures.size());
		}

		//Add the new feature positions into the list of all markers
		int oldSize = allMarkers.numPoints;
		allMarkers.descriptorLength = markerFeatures.descriptorLength;
		allMarkers.numPoints = allMarkers.numPoints + markerFeatures.size();
		allMarkers.features = (CvPoint2D32f*) realloc(allMarkers.features, sizeof(CvPoint2D32f)*allMarkers.numPoints);
		for (int i=oldSize; i<allMarkers.numPoints; i++) {
			allMarkers.features[i] = markerFeatures.at(i-(oldSize)).position;
		}

		//Add the new descriptors
		if (oldSize>0) {
			float *oldDesc = (float*)malloc(sizeof(float)*markerFeatures.descriptorLength*oldSize);
			memcpy(oldDesc, allMarkers.descriptors->ptr<float*>(), sizeof(float)*markerFeatures.descriptorLength*oldSize);
			delete allMarkers.descriptors;
			allMarkers.descriptors = new cv::Mat(allMarkers.numPoints, markerFeatures.descriptorLength, CV_32F);
			memcpy(allMarkers.descriptors->ptr<float*>(), oldDesc, sizeof(float)*markerFeatures.descriptorLength*oldSize);
			free(oldDesc);
		} else {
			allMarkers.descriptors = new cv::Mat(allMarkers.numPoints, markerFeatures.descriptorLength, CV_32F);
		}
		for(int i = oldSize; i < allMarkers.numPoints; i++)
			memcpy(allMarkers.descriptors->ptr<float*>(i), markerFeatures.at(i-(oldSize)).descriptor, markerFeatures.descriptorLength*sizeof(float));

		//Create new marker tree
		if (oldSize>0) 
			delete allMarkers.tree;
		allMarkers.tree = new cv::flann::Index(*(allMarkers.descriptors), cv::flann::KDTreeIndexParams(4));  // using 4 randomized kdtrees
	}

	markerFeatures.clear();

	return true;
}

bool RegistrationAlgorithm::removeMarker(Marker marker) {
	for (unsigned int i=0; i<trainedMarkers.size(); i++) {
		if (trainedMarkers.at(i).markerFilename == marker.name) {
			//Remove single marker
			free(trainedMarkers.at(i).features);
			delete trainedMarkers.at(i).tree;
			delete trainedMarkers.at(i).descriptors;
			trainedMarkers.erase(trainedMarkers.begin()+i);

			//Remove from set of trained markers
			{
				int oldSize = allMarkers.numPoints;
				int featuresBegin = i>0?allMarkersPos.at(i-1):0;
				int featuresEnd = allMarkersPos.at(i);
				int featuresCount= featuresEnd-featuresBegin;
				allMarkers.numPoints = allMarkers.numPoints-featuresCount;

				//Copy additional features across
				if (featuresEnd<oldSize) {
					memcpy(&(allMarkers.features[featuresBegin]), &(allMarkers.features[featuresEnd]), sizeof(CvPoint2D32f)*(oldSize-featuresEnd));
				}
				allMarkers.features = (CvPoint2D32f*)realloc(allMarkers.features, sizeof(CvPoint2D32f)*allMarkers.numPoints);

				//Backup descriptors before and after those being removed
				float *preDesc, *postDesc;
				if (featuresBegin>0) {
					preDesc = (float*)malloc(sizeof(float)*allMarkers.descriptorLength*featuresBegin);
					memcpy(preDesc, allMarkers.descriptors->ptr<float*>(), sizeof(float)*allMarkers.descriptorLength*featuresBegin);
				}
				if (featuresEnd<oldSize) {
					postDesc = (float*)malloc(sizeof(float)*allMarkers.descriptorLength*(oldSize-featuresEnd));
					memcpy(postDesc, allMarkers.descriptors->ptr<float*>(featuresEnd), sizeof(float)*allMarkers.descriptorLength*(oldSize-featuresEnd));
				}

				
				delete allMarkers.descriptors; allMarkers.descriptors = 0;
				delete allMarkers.tree; allMarkers.tree = 0;

				//Create new descriptor matrix
				if (allMarkers.numPoints > 0) allMarkers.descriptors = new cv::Mat(allMarkers.numPoints, allMarkers.descriptorLength, CV_32F);

				//Copy the backups into the new matrix
				if (featuresBegin>0) {
					memcpy(allMarkers.descriptors->ptr<float*>(), preDesc, sizeof(float)*allMarkers.descriptorLength*featuresBegin);
					free(preDesc);
				}
				if (featuresEnd<oldSize) {
					memcpy(allMarkers.descriptors->ptr<float*>(featuresBegin), postDesc, sizeof(float)*allMarkers.descriptorLength*(oldSize-featuresEnd));
					free(postDesc);
				}

				//Create new Marker Tree
				if (allMarkers.numPoints > 0) allMarkers.tree = new cv::flann::Index(*(allMarkers.descriptors), cv::flann::KDTreeIndexParams(4));  // using 4 randomized kdtrees

				//Update the positions
				for (unsigned int j=i; j<allMarkersPos.size(); j++) allMarkersPos.at(j) -= featuresCount;
				allMarkersPos.erase(allMarkersPos.begin()+i);
			}
			return true;

		}
	}
	return false;

}

vector<PointMatches> RegistrationAlgorithm::findAllMatches(IplImage* image) {

	if (trainedMarkers.size() <1) return vector<PointMatches>();

	FeatureVector imageFeatures = performRegistration(image);
	vector<PointMatches> matches = matchAllANN(imageFeatures, 1.1f);
	imageFeatures.clear();

	return matches;
}

PointMatches RegistrationAlgorithm::findMatches(IplImage* image, int index) {

	FeatureVector imageFeatures = performRegistration(image);
	PointMatches matches = matchANN(imageFeatures, 1.3f, trainedMarkers.at(index));
	imageFeatures.clear();
	return matches;
}

RegistrationAlgorithm::TrainedMarker RegistrationAlgorithm::trainMarker(FeatureVector features) {
	//Create and initialise a new marker tree object
	TrainedMarker tm; tm.numPoints = features.size();
	tm.features = (CvPoint2D32f*)malloc(sizeof(CvPoint2D32f)*features.size());
	tm.descriptors = new cv::Mat(features.size(), features.descriptorLength, CV_32F);
	tm.descriptorLength = features.descriptorLength;

	//If there's no points don't worry about setting anything up
	if (features.size()==0) return tm;
	
    //Populate the tree points and point positions
	for(unsigned int i = 0; i < features.size(); i++ )
    {
		memcpy(tm.descriptors->ptr<float*>(i), features.at(i).descriptor, features.descriptorLength*sizeof(float));
		tm.features[i] = features.at(i).position; 
    }

	//Initialise the tree
	tm.tree = new cv::flann::Index(*(tm.descriptors), cv::flann::KDTreeIndexParams(4));  // using 4 randomized kdtrees

	return tm;
}

PointMatches RegistrationAlgorithm::matchANN(FeatureVector features, float thresh, TrainedMarker tm) {
	//Initialise matrices for the features, indices and distances
	cv::Mat m_features(features.size(), features.descriptorLength, CV_32F);
	cv::Mat m_indices(features.size(), 2, CV_32S);
    cv::Mat m_dists(features.size(), 2, CV_32F);

	//Create and initialise the returning point match object
	PointMatches matches;

	//If there's no features, just return an empty structure
	if(features.size()==0) return matches;

	//Initialise feature search set
    for(unsigned int i = 0; i < features.size(); i++ )
    {
		memcpy(m_features.ptr<float*>(i), features.at(i).descriptor, features.descriptorLength*sizeof(float));
    }

	//Perform search
	tm.tree->knnSearch(m_features, m_indices, m_dists, 2, cv::flann::SearchParams(features.descriptorLength)); // maximum number of leafs checked

	//Keep a record of the minimum distance for each marker feature
	float *minDistances = new float[tm.numPoints]; for (int i=0; i<tm.numPoints; i++) minDistances[i] = -1;

	//Keep a track of the matches
	int *fMatches = new int[features.size()]; 
	
	//Keep a track of how many features we've found, and how many were invalid
	int foundMatches=0; int invalidMatches=0;

	//Search through each feature
	for (unsigned int i=0; i<features.size(); i++) {
		//If the best match distance * thresh is less than the second best match
		if (m_dists.at<float>(i,0)*thresh <=m_dists.at<float>(i,1)) { 

			//If the minDistance isn't initialised set it up
			if (minDistances[m_indices.at<int>(i,0)] == -1.f) {
				fMatches[foundMatches] = i; foundMatches++; 
				minDistances[m_indices.at<int>(i,0)] = m_dists.at<float>(i,0);
			//Otherwise copy the new one across and keep a count of the over written one
			} else if (minDistances[m_indices.at<int>(i,0)] > m_dists.at<float>(i,0)) {
				fMatches[foundMatches] = i; foundMatches++; 
				minDistances[m_indices.at<int>(i,0)] = m_dists.at<float>(i,0);
				invalidMatches++;
			}
		}
	}

	//Initialise the returning point match object
	matches.resize(foundMatches-invalidMatches);

	//Loop through
	int currentMatch = 0;
	for (int i=0; i<foundMatches; i++) {
		//If this is the best possible match copy it across
		if (m_dists.at<float>(fMatches[i],0) == minDistances[m_indices.at<int>(fMatches[i],0)]) {
			matches.featMarker[currentMatch] =  tm.features[m_indices.at<int>(fMatches[i],0)];
			matches.featImage[currentMatch] = features.at(fMatches[i]).position;
			matches.score[currentMatch] = m_dists.at<float>(fMatches[i],0);
			currentMatch++;
		}
	}

	//Clean up
	free(minDistances); free(fMatches);
	m_features.release(); m_indices.release(); m_dists.release();

	return matches;
}

int RegistrationAlgorithm::findMarkerIndex(int featureIndex) {
	for (unsigned int j = 0; j<allMarkersPos.size();j++) {
		if (featureIndex<=allMarkersPos.at(j)) {
			return j;
		}
	}
	return -1;
}

vector<PointMatches> RegistrationAlgorithm::matchAllANN(FeatureVector features, float thresh) {
	//Initialise matrices for the features, indices and distances
	cv::Mat m_features(features.size(), features.descriptorLength, CV_32F);
	cv::Mat m_indices(features.size(), 2, CV_32S);
    cv::Mat m_dists(features.size(), 2, CV_32F);

	//Create and initialise the returning point match object
	int markerCount = allMarkersPos.size();
	vector<PointMatches> allMatches; allMatches.resize(markerCount);

	//If there's no features, just return an empty structure
	if(features.size()==0) return allMatches;

	//Initialise feature search set
    for(unsigned int i = 0; i < features.size(); i++ )
    {
		memcpy(m_features.ptr<float*>(i), features.at(i).descriptor, features.descriptorLength*sizeof(float));
    }

	//Perform search
	allMarkers.tree->knnSearch(m_features, m_indices, m_dists, 2, cv::flann::SearchParams(features.descriptorLength)); // maximum number of leafs checked

	//Keep a record of the minimum distance for each marker feature
	float *minDistances = new float[allMarkers.numPoints]; for (int i=0; i<allMarkers.numPoints; i++) minDistances[i] = -1;

	//Keep a track of the matches
	int *fMatches = new int[features.size()]; 

	//Keep a track of how many features we've found, and how many were invalid
	int *foundMatches = (int*)calloc(markerCount,sizeof(int)); int *invalidMatches= (int*)calloc(markerCount,sizeof(int));
	
	//Keep a track of the total number of matches;
	int totalMatches = 0;

	//Search through each feature
	for (unsigned int i=0; i<features.size(); i++) {
		//If the best match distance * thresh is less than the second best match
		if (m_dists.at<float>(i,0)*thresh <=m_dists.at<float>(i,1)) { 
			//Get the Marker Index
			int markerIndex = findMarkerIndex(m_indices.at<int>(i,0));

			//If the minDistance isn't initialised set it up
			if (minDistances[m_indices.at<int>(i,0)] == -1.f) {
				fMatches[totalMatches] = i; foundMatches[markerIndex]++; totalMatches++; 
				minDistances[m_indices.at<int>(i,0)] = m_dists.at<float>(i,0);
			//Otherwise copy the new one across and keep a count of the over written one
			} else if (minDistances[m_indices.at<int>(i,0)] > m_dists.at<float>(i,0)) {
				fMatches[totalMatches] = i; foundMatches[markerIndex]++; totalMatches++; 
				minDistances[m_indices.at<int>(i,0)] = m_dists.at<float>(i,0);
				invalidMatches[markerIndex]++;
			}
		}
	}

	//Initialise the returning point match object
	for (unsigned int i=0; i<markerCount; i++) {
		allMatches.at(i).resize(foundMatches[i]-invalidMatches[i]);
	}

	int *curMatchCount = (int*)calloc(markerCount, sizeof(int));

	//Loop through and clean up
	for (int i=0; i<totalMatches; i++) {
		if (m_dists.at<float>(fMatches[i],0) == minDistances[m_indices.at<int>(fMatches[i],0)]) {
			int markerIndex = findMarkerIndex(m_indices.at<int>(fMatches[i],0));
			allMatches.at(markerIndex).featMarker[curMatchCount[markerIndex]] = allMarkers.features[m_indices.at<int>(fMatches[i],0)];
			allMatches.at(markerIndex).featImage[curMatchCount[markerIndex]] = features.at(fMatches[i]).position;
			allMatches.at(markerIndex).score[curMatchCount[markerIndex]] = m_dists.at<float>(fMatches[i],0);
			curMatchCount[markerIndex]++;
		}
	}

	//Clean up
	free(curMatchCount); 
	free(minDistances); free(fMatches);
	free(foundMatches); free(invalidMatches);

	m_features.release(); m_indices.release(); m_dists.release();

	return allMatches;
}