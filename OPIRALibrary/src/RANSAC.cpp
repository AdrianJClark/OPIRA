//
// file : homographyEst.cpp
//------------------------------------------------
// this file contains functions for homography
// estimation
//

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
#include "OPIRAlibrary.h"

#define EPS 0.5
#define T_SQUARE 100 // t = sqrt(6)* sigma and set sigma = sqrt(6)

using namespace OPIRALibrary;

void CalculateDistance(CvMat* H, PointMatches* corspMap, PointMatches* inlierMap);
bool IsGoodSample(CvPoint2D32f* points, int numOfPoints);
bool IsColinear(CvMat* A, CvMat* B, CvMat* C);

PointMatches OPIRALibrary::Ransac2(PointMatches corspMap)
{
	PointMatches inlierMap;

	if (corspMap.count>3) {
		CvMat markerFeatures = cvMat(1, corspMap.count, CV_32FC2, corspMap.featMarker);
		CvMat imageFeatures = cvMat(1, corspMap.count, CV_32FC2, corspMap.featImage);

		cvFindHomography(&markerFeatures, &imageFeatures, inlierMap.homography, CV_LMEDS, .99);
		CalculateDistance(inlierMap.homography, &corspMap, &inlierMap);
	}
	return inlierMap;

}

PointMatches OPIRALibrary::Ransac(PointMatches corspMap)
{
	const int numOfCorresp = 4;
	CvPoint2D32f domainPositions[numOfCorresp];
	CvPoint2D32f rangePositions[numOfCorresp];

	int i, pos; 

	int numOfInliers;
	int totalNumOfPoints = corspMap.count;
	int maxNumOfInliers = 1;

	PointMatches inlierMap;
	if (corspMap.count < numOfCorresp) return inlierMap;

	CvMat* Htmp = cvCreateMat(3, 3, CV_32FC1);

	float p = 0.99f;
	float e = 0.5f;
	float N = 1000.f;
	float outlierProb;

	int badCount = 0;
	int sampleCount = 0;

	while( (float)sampleCount < N && badCount < 50 )
	{
		// pick 4 corresponding points
		for( i = 0 ; i < numOfCorresp ; i++ )
		{
			pos = rand() % corspMap.count; // select random positions
			
			rangePositions[i]  = corspMap.featImage[pos];
			domainPositions[i] = corspMap.featMarker[pos];
		}

		// check whether samples are good or not.
		// if the selected samples are good, then do homography estimation
		// else reselect samples.
		if( IsGoodSample(domainPositions, numOfCorresp) &&
			IsGoodSample(rangePositions, numOfCorresp) )
		{
			PointMatches tempInlierMap; tempInlierMap.count=0;

			cvGetPerspectiveTransform(domainPositions, rangePositions, Htmp);

			CalculateDistance(Htmp, &corspMap, &tempInlierMap);

			// choose H with the largest number of inliears
			numOfInliers = tempInlierMap.count;
			if( numOfInliers >= maxNumOfInliers )
			{
				maxNumOfInliers = numOfInliers;
				inlierMap.clone(tempInlierMap);
			}

			tempInlierMap.clear();

			// adaptive algorithm for determining the number of RANSAC samples
			// textbook algorithm 4.6
			
			totalNumOfPoints = corspMap.count;
			outlierProb = 1 - ((float)maxNumOfInliers / (float)totalNumOfPoints);
			e = (e < outlierProb) ? e : outlierProb;
			N = log(1 - p) / log(1 - pow((1 - e), numOfCorresp));
			sampleCount++;
		}
		else
		{
			badCount++;
		}
	}

	cvReleaseMat(&Htmp);

	return inlierMap;
}


//
// function : CalculateDistance
// usage : CalculateDistance(H, corspMap, inlierMap);
// ---------------------------------------------------
// This function calculates distance of data using
// symmetric transfer error. Then, compute inliers
// that consist with H.
//
void CalculateDistance(CvMat *H, PointMatches *corspMap, PointMatches *inlierMap)
{
	int i;
	float x1, y1, x2, y2;
	double x1Trans, y1Trans, w1Trans, x2Trans, y2Trans, w2Trans;
	double dist2x1AndInvHx2, dist2x2AndHx1, dist2Trans;
	
	CvMat* invH = cvCreateMat(3, 3, CV_32FC1);
	cvInvert(H, invH);

	PointMatches tmpInlier; tmpInlier.resize(corspMap->count); tmpInlier.count = 0; cvCopy(H, tmpInlier.homography,0);

	// use d^2_transfer as distance measure
	for( i = 0 ; i < corspMap->count; i++ )
	{
		x1 = corspMap->featMarker[i].x;
		y1 = corspMap->featMarker[i].y;
		x2 = corspMap->featImage[i].x;
		y2 = corspMap->featImage[i].y;

		// calculate x_trans = H * x
		x2Trans = cvmGet(H, 0, 0) * x1 + cvmGet(H, 0, 1) * y1 + cvmGet(H, 0, 2);
		y2Trans = cvmGet(H, 1, 0) * x1 + cvmGet(H, 1, 1) * y1 + cvmGet(H, 1, 2);
		w2Trans = cvmGet(H, 2, 0) * x1 + cvmGet(H, 2, 1) * y1 + cvmGet(H, 2, 2);
		x2Trans = x2Trans / w2Trans;
		y2Trans = y2Trans / w2Trans;

		// calculate x'_trans = H^(-1) * x'
		x1Trans = cvmGet(invH, 0, 0) * x2 + cvmGet(invH, 0, 1) * y2 + cvmGet(invH, 0, 2);
		y1Trans = cvmGet(invH, 1, 0) * x2 + cvmGet(invH, 1, 1) * y2 + cvmGet(invH, 1, 2);
		w1Trans = cvmGet(invH, 2, 0) * x2 + cvmGet(invH, 2, 1) * y2 + cvmGet(invH, 2, 2);
		x1Trans = x1Trans / w1Trans;
		y1Trans = y1Trans / w1Trans;

		// calculate the square distance (symmetric transfer error)
		dist2x1AndInvHx2 = (x1 - x1Trans)*(x1 - x1Trans) + (y1 - y1Trans)*(y1 - y1Trans);
		dist2x2AndHx1 = (x2 - x2Trans)*(x2 - x2Trans) + (y2 - y2Trans)*(y2 - y2Trans);
		dist2Trans = dist2x1AndInvHx2 + dist2x2AndHx1;
		
		if( dist2Trans < T_SQUARE ) {
			tmpInlier.featMarker[tmpInlier.count] = cvPoint2D32f(x1,y1);
			tmpInlier.featImage[tmpInlier.count] = cvPoint2D32f(x2,y2);
			tmpInlier.count++;
		}
	}

	inlierMap->clone(tmpInlier);
	tmpInlier.clear();

	// release matrices
	cvReleaseMat(&invH);
}

//
// function : IsGoodSample
// usage : r = IsGoodSample(points, numOfPoints)
// -------------------------------------------------
// This function checks colinearity of all given points.
//
bool IsGoodSample(CvPoint2D32f* points, int numOfPoints)
{
	bool ret = false;
	int i, j, k;

	CvMat* A = cvCreateMat(3, 1, CV_32FC1);
	CvMat* B = cvCreateMat(3, 1, CV_32FC1);
	CvMat* C = cvCreateMat(3, 1, CV_32FC1);

	i = 0;
	j = i + 1;
	k = j + 1;

	// check colinearity recursively
	while(true)
	{
		// set point vectors
		cvmSet(A, 0, 0, points[i].x);
		cvmSet(A, 1, 0, points[i].y);
		cvmSet(A, 2, 0, 1);
		cvmSet(B, 0, 0, points[j].x);
		cvmSet(B, 1, 0, points[j].y);
		cvmSet(B, 2, 0, 1);
		cvmSet(C, 0, 0, points[k].x);
		cvmSet(C, 1, 0, points[k].y);
		cvmSet(C, 2, 0, 1);

		// check linearity
		ret = IsColinear(A, B, C) || ret;

		// update point index
		if( k < numOfPoints - 1 )
		{
			k += 1;
		}
		else
		{
			if( j < numOfPoints - 2 )
			{
				j += 1;
				k = j + 1;
			}
			else
			{
				if( i < numOfPoints - 3 )
				{
					i += 1;
					j = i + 1;
					k = j + 1;
				}
				else
				{
					break;
				}
			}
		}
	}

	cvReleaseMat(&A);
	cvReleaseMat(&B);
	cvReleaseMat(&C);

	return(!ret);
}

//
// function : IsColinear
// usage : r = IsColinear(A, B, C);
// --------------------------------------
// This function checks the colinearity of
// the given 3 points A, B, and C.
// If these are colinear, it returns false. (true 반환해야하는거 아냐?)
//
bool IsColinear(CvMat *A, CvMat *B, CvMat *C)
{
	bool ret = false;

	CvMat* lineAB = cvCreateMat(3, 1, CV_32FC1);
	cvCrossProduct(A, B, lineAB);

	double d = cvDotProduct(lineAB, C);

	if( (d < EPS) && (d > -EPS) )
		ret = true;

	// release matrices
	cvReleaseMat(&lineAB);

	return ret;
}