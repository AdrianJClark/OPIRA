#include "OPIRALibraryMT.h"
#include "opencv/highgui.h"

using namespace OPIRALibrary;
using namespace std;

Mutex::Mutex() { mutex = false; }
Mutex::~Mutex() { }
void Mutex::acquire() { while (mutex) Sleep(1); mutex = true; }
void Mutex::release() { mutex = false; }

DWORD WINAPI run( LPVOID lpParam ) {
	RegistrationThread::ThreadData *t= (RegistrationThread::ThreadData*)lpParam;

	while (t->running) {
		if (t->updated) {
			t->newMutex.acquire();	
				vector<PointMatches> newMatches = t->regAlgorithm->findAllMatches(t->newFrame);
			t->newMutex.release();

			t->curMutex.acquire(); 
				for (unsigned int i=0; i<t->curMatches->size(); i++) t->curMatches->at(i).clear(); t->curMatches->clear(); 
				for (unsigned int i=0; i<newMatches.size(); i++) t->curMatches->push_back(Ransac(newMatches.at(i)));
				t->newMutex.acquire(); 
					cvReleaseImage(&t->curFrame); t->curFrame = cvCloneImage(t->newFrame); 
					cvReleaseImage(&t->newFrame); t->newFrame = 0;
				t->newMutex.release();
			t->curMutex.release();
			
			for (unsigned int i=0; i<newMatches.size(); i++) newMatches.at(i).clear(); newMatches.clear();
			t->updated = false;
		} else {
			Sleep(1);
		}
	}
	t->finished = true;
	return 0;
};

RegistrationThread::RegistrationThread(RegistrationAlgorithm *registrationAlgorithm) {
	tData = (ThreadData*)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(ThreadData));
	
	tData->regAlgorithm = registrationAlgorithm;
	tData->running = true; tData->updated = false; tData->newFrame = tData->curFrame = 0; tData->finished = false;
	tData->curMatches = new vector<PointMatches>();
	DWORD dwThreadId;
	CreateThread(NULL, 0, &run, tData,0,&dwThreadId);
}

RegistrationThread::~RegistrationThread() {
	deleteReg();
	for (unsigned int i=0; i<tData->curMatches->size(); i++) tData->curMatches->at(i).clear(); tData->curMatches->clear();
	delete tData->curMatches;
	if (tData->curFrame) cvReleaseImage(&tData->curFrame); if (tData->newFrame) cvReleaseImage(&tData->newFrame);
	HeapFree(GetProcessHeap(), 0, tData);
}

void RegistrationThread::deleteReg() {
	if (tData->regAlgorithm) delete tData->regAlgorithm;
	tData->regAlgorithm = 0;
}

bool RegistrationThread::isFinished() {
	return tData->finished;
}

void RegistrationThread::stop() {
	tData->running=false;
	while (!tData->finished) Sleep(1);
}

vector<PointMatches> RegistrationThread::findAllMatches(IplImage *frame_input) {
	vector<PointMatches> returnMatches; 

	if (!tData->updated) {
		tData->newMutex.acquire(); 
			tData->newFrame = cvCloneImage(frame_input); tData->updated = true;
		tData->newMutex.release(); 
	}

	if (tData->curFrame) {
		tData->curMutex.acquire(); 
		returnMatches = allOpticalFlow(tData->curFrame, *tData->curMatches, frame_input);
		tData->curMutex.release();
	}

	return returnMatches;
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
	tData->newMutex.acquire();	
		retVal = tData->regAlgorithm->removeMarker(marker);
	tData->newMutex.release();
	return retVal;
}

bool RegistrationThread::addMarker(Marker marker) {
	bool retVal;
	tData->newMutex.acquire();	
		retVal = tData->regAlgorithm->addMarker(marker);
	tData->newMutex.release();
	return retVal;
}
