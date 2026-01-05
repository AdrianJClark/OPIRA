#include "OPIRALibraryMT-ZThread.h"
#include "highgui.h"

using namespace OPIRALibrary;

RegistrationThread::RegistrationThread(RegistrationAlgorithm *registrationAlgorithm) {
	regAlgorithm = registrationAlgorithm;
	running = true; updated = false; newFrame = curFrame = 0; finished = false;
}

RegistrationThread::~RegistrationThread() {
	deleteReg();
	for (unsigned int i=0; i<curMatches.size(); i++) curMatches.at(i).clear();
	if (curFrame) cvReleaseImage(&curFrame); if (newFrame) cvReleaseImage(&newFrame);
	finished = true;
}

void RegistrationThread::deleteReg() {
	if (regAlgorithm) delete regAlgorithm;
	regAlgorithm = 0;
}

bool RegistrationThread::isFinished() {
	return finished;
}

void RegistrationThread::run() {
	while (running) {
		if (updated) {
			newMutex.acquire();	
				vector<PointMatches> newMatches = regAlgorithm->findAllMatches(newFrame);
			newMutex.release();
			
			curMutex.acquire(); 
				for (unsigned int i=0; i<curMatches.size(); i++) curMatches.at(i).clear(); curMatches.clear(); 
				for (unsigned int i=0; i<newMatches.size(); i++) curMatches.push_back(Ransac(newMatches.at(i)));
				newMutex.acquire(); 
					cvReleaseImage(&curFrame); curFrame = cvCloneImage(newFrame); 
					cvReleaseImage(&newFrame); newFrame = 0;
				newMutex.release();
			curMutex.release();

			for (unsigned int i=0; i<newMatches.size(); i++) newMatches.at(i).clear(); newMatches.clear(); 
			updated = false;
		} else {
			Thread::sleep(1);
		}
	}
};

void RegistrationThread::stop() {
	running=false;
}

vector<PointMatches> RegistrationThread::findAllMatches(IplImage *frame_input) {
	vector<PointMatches> returnMatches; 

	if (!updated) {
		newMutex.acquire(); 
			newFrame = cvCloneImage(frame_input); updated = true;
		newMutex.release(); 
	}

	if (curFrame) {
		/*curMutex.acquire(); 
		vector<PointMatches> m; m.resize(curMatches.size()); 
		for (unsigned int i=0; i<curMatches.size(); i++) m.at(i).clone(curMatches.at(i));
		IplImage *f = cvCloneImage(curFrame);
		curMutex.release(); 

		//Calculate the latest position for all the matches
		for(unsigned int i=0; i<m.size(); i++) {
			returnMatches.push_back(opticalFlow(f, m.at(i), frame_input));
			m.at(i).clear();
		}

		//Clean up
		cvReleaseImage(&f); m.clear();*/
		
		curMutex.acquire(); 
		returnMatches = allOpticalFlow(curFrame, curMatches, frame_input);
		curMutex.release();
	}

	return returnMatches;
}




PointMatches RegistrationThread::opticalFlow(IplImage *previousImage, PointMatches prevMatches, IplImage *currentImage) {
	if (prevMatches.count==0) return PointMatches();
	if (previousImage->width != currentImage->width || previousImage->height != currentImage->height)  return PointMatches();

	CvPoint2D32f *currentFeatures;
	PointMatches flowMatches;

	IplImage *grayPrev_image= cvCreateImage(cvGetSize(previousImage), IPL_DEPTH_8U, 1); 
	IplImage *grayCurrent_image= cvCreateImage(cvGetSize(currentImage), IPL_DEPTH_8U, 1);
	if (previousImage->nChannels >1) { cvConvertImage(previousImage, grayPrev_image); } else { cvCopy(previousImage, grayPrev_image); }
	if (currentImage->nChannels >1) { cvConvertImage(currentImage, grayCurrent_image); } else { cvCopy(currentImage, grayCurrent_image); }

	//Allocate Space for GoodPoint Registers
	currentFeatures = (CvPoint2D32f*)cvAlloc(prevMatches.count*sizeof(CvPoint2D32f));

	char* status = (char *)malloc(prevMatches.count*sizeof(char));

	cvCalcOpticalFlowPyrLK(grayPrev_image, grayCurrent_image, 0, 0, (CvPoint2D32f *)prevMatches.featImage, currentFeatures, prevMatches.count, cvSize(10,10), 3, status, 0, cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03), 0 );

	flowMatches.resize(prevMatches.count); flowMatches.count =0;
	for (int i=0; i<prevMatches.count; i++) {
		if(status[i]==1) {
			flowMatches.featMarker[flowMatches.count].x = (float)cvRound(prevMatches.featMarker[i].x); flowMatches.featMarker[flowMatches.count].y = (float)cvRound(prevMatches.featMarker[i].y);
			flowMatches.featImage[flowMatches.count].x = (float)cvRound(currentFeatures[i].x); flowMatches.featImage[flowMatches.count].y = (float)cvRound(currentFeatures[i].y);
			flowMatches.count++;
		}
	}

	cvReleaseImage(&grayPrev_image);
	cvReleaseImage(&grayCurrent_image);
	cvFree((void**)&currentFeatures);
	free(status);
	
	return flowMatches;
}

vector<PointMatches> RegistrationThread::allOpticalFlow(IplImage *previousImage, vector<PointMatches> prevMatches, IplImage *currentImage) {
	if (previousImage->width != currentImage->width || previousImage->height != currentImage->height) {
		vector<PointMatches> flowMatches;
		for (int i=0; i<prevMatches.size(); i++) { PointMatches p; p.count = 0; flowMatches.push_back(p); }
		return flowMatches;
	}

	//Set up Images
	IplImage *grayPrev_image= cvCreateImage(cvGetSize(previousImage), IPL_DEPTH_8U, 1); 
	IplImage *grayCurrent_image= cvCreateImage(cvGetSize(currentImage), IPL_DEPTH_8U, 1);
	if (previousImage->nChannels >1) { cvConvertImage(previousImage, grayPrev_image); } else { cvCopy(previousImage, grayPrev_image); }
	if (currentImage->nChannels >1) { cvConvertImage(currentImage, grayCurrent_image); } else { cvCopy(currentImage, grayCurrent_image); }
	
	//Calculate Total Point Count
	int totalCount = 0; for (unsigned int i=0; i<prevMatches.size(); i++) totalCount+=prevMatches.at(i).count;
	
	//Initialise Previous Features
	CvPoint2D32f *allPrevFeatures = (CvPoint2D32f*)cvAlloc(totalCount*sizeof(CvPoint2D32f));
	int curPos=0; for (unsigned int i=0; i<prevMatches.size(); i++) {
		int size = prevMatches.at(i).count * sizeof(CvPoint2D32f); 
		memcpy(allPrevFeatures+curPos, prevMatches.at(i).featImage, size); 
		curPos+= prevMatches.at(i).count; }

	//Allocate Space for GoodPoint Registers
	CvPoint2D32f *allCurrentFeatures = (CvPoint2D32f*)cvAlloc(totalCount*sizeof(CvPoint2D32f));
	char* status = (char *)malloc(totalCount*sizeof(char));

	cvCalcOpticalFlowPyrLK(grayPrev_image, grayCurrent_image, 0, 0, allPrevFeatures, allCurrentFeatures, totalCount, cvSize(10,10), 3, status, 0, cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03), 0 );

	//Initialize the returned vector
	vector<PointMatches> flowMatches; flowMatches.resize(prevMatches.size()); curPos = 0; int lastIndex =0;
	
	for (int i=0; i<prevMatches.size(); i++) {
		flowMatches.at(i).resize(prevMatches.at(i).count); flowMatches.at(i).count =0;
		for (int j=0; j<prevMatches.at(i).count; j++) {
			if(status[curPos]==1) {
				flowMatches.at(i).featMarker[flowMatches.at(i).count].x = (float)cvRound(prevMatches.at(i).featMarker[j].x); flowMatches.at(i).featMarker[flowMatches.at(i).count].y = (float)cvRound(prevMatches.at(i).featMarker[j].y);
				flowMatches.at(i).featImage[flowMatches.at(i).count].x = (float)cvRound(allCurrentFeatures[curPos].x); flowMatches.at(i).featImage[flowMatches.at(i).count].y = (float)cvRound(allCurrentFeatures[curPos].y);
				flowMatches.at(i).count++;	curPos++;
			}
		}
	}

	cvReleaseImage(&grayPrev_image);
	cvReleaseImage(&grayCurrent_image);
	cvFree((void**)&allCurrentFeatures); cvFree((void**)&allPrevFeatures);
	free(status);
	
	return flowMatches;
}

bool RegistrationThread::removeMarker(Marker marker) {
	bool retVal;
	newMutex.acquire();	
		retVal = regAlgorithm->removeMarker(marker);
	newMutex.release();
	return retVal;
}

bool RegistrationThread::addMarker(Marker marker) {
	bool retVal;
	newMutex.acquire();	
		retVal = regAlgorithm->addMarker(marker);
	newMutex.release();
	return retVal;
}