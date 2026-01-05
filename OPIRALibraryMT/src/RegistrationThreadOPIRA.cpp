#include "OPIRALibraryMT.h"
#include "opencv/highgui.h"

using namespace OPIRALibrary;

DWORD WINAPI opirarun( LPVOID lpParam ) {
	RegistrationThreadOPIRA::ThreadOPIRAData *t= (RegistrationThreadOPIRA::ThreadOPIRAData*)lpParam;

	while (t->running) {
		if (t->updated) {
			t->newMutex.acquire();	
				IplImage *unDistort = cvCreateImage(t->mSize, t->newFrame->depth, t->newFrame->nChannels);
				cvWarpPerspective(t->newFrame, unDistort, t->newHomography, 1+8+16);
				PointMatches newMatches =  t->regAlgorithm->findMatches(unDistort, 0);
				cvReleaseImage(&unDistort);
				if (newMatches.count>0) {
					CvMat mMatches = cvMat(newMatches.count, 1, CV_32FC2, newMatches.featImage);
					cvPerspectiveTransform(&mMatches, &mMatches, t->newHomography);
				}
			t->newMutex.release();


			t->curMutex.acquire(); 
				t->curMatches.clear();
				t->curMatches = Ransac(newMatches);
				t->newMutex.acquire(); 
					cvReleaseImage(&t->curFrame); t->curFrame = cvCloneImage(t->newFrame); 
					cvReleaseImage(&t->newFrame); t->newFrame = 0;
					cvReleaseMat(&t->newHomography); t->newHomography = 0;
				t->newMutex.release();
			t->curMutex.release();

			newMatches.clear(); 
			t->updated = false;
		} else {
			Sleep(1);
		}
	}
	t->finished = true;
	return 0;
};

RegistrationThreadOPIRA::RegistrationThreadOPIRA(RegistrationAlgorithm *registrationAlgorithm, CvSize markerSize) {
	tData = (ThreadOPIRAData*)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(ThreadOPIRAData));
	
	tData->regAlgorithm = registrationAlgorithm;
	tData->running = true; tData->updated = false; tData->newFrame = tData->curFrame = 0; tData->finished = false;
	tData->mSize = markerSize; tData->newHomography = 0;

	DWORD dwThreadId;
	CreateThread(NULL, 0, &opirarun, tData,0,&dwThreadId);
}

RegistrationThreadOPIRA::~RegistrationThreadOPIRA() {
	if (tData->regAlgorithm) delete tData->regAlgorithm;
	if (tData->curFrame) cvReleaseImage(&tData->curFrame); 
	if (tData->newFrame) cvReleaseImage(&tData->newFrame);
	if (tData->newHomography) cvReleaseMat(&tData->newHomography);
	tData->curMatches.clear();
	HeapFree(GetProcessHeap(), 0, tData);
}

void RegistrationThreadOPIRA::deleteReg() {
	if (tData->regAlgorithm) delete tData->regAlgorithm;
	tData->regAlgorithm = 0;
}

bool RegistrationThreadOPIRA::isFinished() {
	return tData->finished;
}

void RegistrationThreadOPIRA::stop() {
	tData->running=false;
	while (!tData->finished) Sleep(1);
}


PointMatches RegistrationThreadOPIRA::undistortRegister(IplImage *frame_input, CvMat *homography, CvSize frameSize) {
	PointMatches returnMatches; 

	if (!tData->updated) {
		tData->newMutex.acquire(); 
			tData->newFrame = cvCloneImage(frame_input); tData->newHomography = cvCloneMat(homography); tData->updated = true;
			tData->mSize = frameSize;
		tData->newMutex.release(); 
	}

	if (tData->curFrame) {
		//Copy the last known good registration
		tData->curMutex.acquire(); 
		PointMatches m; m.clone(tData->curMatches);
		IplImage *f = cvCloneImage(tData->curFrame);
		tData->curMutex.release(); 

		//Calculate the latest position for all the matches
		returnMatches.clear();
		returnMatches = opticalFlow(f, m, frame_input);
		m.clear();

		//Clean up
		cvReleaseImage(&f); m.clear();
	}

	return returnMatches;
}

PointMatches RegistrationThreadOPIRA::opticalFlow(IplImage *previousImage, PointMatches prevMatches, IplImage *currentImage) {
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

bool RegistrationThreadOPIRA::removeMarker(Marker marker) {
	bool retVal;
	tData->newMutex.acquire();	
		retVal = tData->regAlgorithm->removeMarker(marker);
	tData->newMutex.release();
	return retVal;
}

bool RegistrationThreadOPIRA::addMarker(Marker marker) {
	bool retVal;
	tData->newMutex.acquire();	
		retVal = tData->regAlgorithm->addMarker(marker);
	tData->newMutex.release();
	return retVal;
}
