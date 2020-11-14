#pragma once
#define MAX_ACTIVE_FRAMES 100

#include "settings.h"
#include "NumType.h"
#include <iostream>
#include <fstream>
#include "globalCalib.h"

#ifndef _SFRSS_FRAME
#define _SFRSS_FRAME
namespace SFRSS
{

struct FramePym
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	Eigen::Vector3f* dI;			    // dI 保存有3个channel, 第一个grep图，后两个是纵向和横向的difference
	Eigen::Vector3f* dIp[PYR_LEVELS];   // intensity, dx, dy
	float* absSquaredGrad[PYR_LEVELS];  // only used for pixel select (histograms etc.). no NAN.
	float * mColor;
	uint8_t * grayPyr[PYR_LEVELS];
	uint8_t * grayX2;

	inline ~FramePym()
	{
		for(int i = 0; i < PyrLevelsUsedG; i++){
			if (dIp[i]) delete[] dIp[i];
			if (absSquaredGrad[i]) delete[] absSquaredGrad[i];
			if (grayPyr[i]) delete[] grayPyr[i];
		}

		if (mColor != NULL){
			delete[] mColor;
			mColor = NULL;
		}

		if (grayX2 == NULL){
			delete[] grayX2;
		}
	};
	
	inline FramePym(){
		mColor = NULL;
		grayX2 = NULL;
		for (int lvl = 0; lvl < PyrLevelsUsedG; ++lvl){
			dIp[lvl] = NULL;
			absSquaredGrad[lvl] = NULL;
			grayPyr[lvl] = NULL;
		}
	};

    void makeImages(uint8_t * color);
	void setX2Resolution(uint8_t * color);
};

}

#endif