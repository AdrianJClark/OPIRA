#include "OPIRALibrary.h"

using namespace OPIRALibrary;

PointMatches::PointMatches(): featMarker(0), featImage(0), score(0), count(0), homography(0) { 
	//homography = cvCreateMat(3,3,CV_32FC1); 
}

PointMatches::~PointMatches() {
//	clear();
}

void PointMatches::resize(int size) {
	clear(); if (size==0) return;
	featMarker=(CvPoint2D32f*)malloc(size*sizeof(CvPoint2D32f));
	featImage=(CvPoint2D32f*)malloc(size*sizeof(CvPoint2D32f));
	score=(float*)malloc(size*sizeof(float));
	homography = cvCreateMat(3,3,CV_32FC1);
	count=size;
} 
		
void PointMatches::clear() {
	if (featMarker!=0) free(featMarker); featMarker=0;
	if (featImage!=0) free(featImage); featImage=0;
	if (score!=0) free(score); score=0;
	if (homography!=0) cvReleaseMat(&homography); homography = 0;
	count=0;
}

void PointMatches::clone(PointMatches src)
{
	resize(src.count); if (src.count==0) return;
	cvCopy(src.homography, homography);
	memcpy(featImage, src.featImage, src.count*sizeof(CvPoint2D32f));
	memcpy(featMarker, src.featMarker, src.count*sizeof(CvPoint2D32f));
	memcpy(score, src.score, src.count*sizeof(float));
}
