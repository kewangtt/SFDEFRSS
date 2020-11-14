/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/



#pragma once
#include "settings.h"
#include "NumType.h"
#include "string.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#ifndef GLOBAL_CALIB
#define GLOBAL_CALIB
#define THREAD_CNT 8

namespace SFRSS
{
	extern int wG[PYR_LEVELS], hG[PYR_LEVELS];
	extern double fxG[PYR_LEVELS], fyG[PYR_LEVELS],
		  cxG[PYR_LEVELS], cyG[PYR_LEVELS];

	extern double fxiG[PYR_LEVELS], fyiG[PYR_LEVELS],
		  cxiG[PYR_LEVELS], cyiG[PYR_LEVELS];

	extern Mat33 KG[PYR_LEVELS],KiG[PYR_LEVELS];
	extern cv::Mat cvKG[PYR_LEVELS],cvKiG[PYR_LEVELS];

	extern cv::Mat cvK0G, cvK0iG;
	extern cv::Mat cvK1G, cvK1iG;
	extern cv::Mat cvDistCoef0, cvDistCoef1;

	extern Mat33 RectLeftRG;

	extern int rawColsG, rawRowsG;
	extern int baselineScaleG;

	// extern int DispSizeG;
	extern int invalidDispG;
	extern int subPixelLevlG;
	extern int maxDisparityG[PYR_LEVELS];

	extern float wM3G;
	extern float hM3G;
	extern double bfxG;
	extern double bfxG2;
	extern double baselineG;

	extern double RowTimeG;
	extern int PyrLevelsUsedG;

	extern Mat33 RLRG;
	extern Vec3 TLRG;
	extern cv::Mat cvRLRG, cvTLRG;

	extern cv::Mat M1lG, M2lG, M1rG, M2rG;
	extern cv::Mat unM1lG, unM2lG, unM1rG, unM2rG;
	extern cv::Mat unM1rx2G, unM2rx2G;

	extern float * ptrLeftRemapXG;
	extern float * ptrLeftRemapYG;
	extern float * ptrRightRemapXG;
	extern float * ptrRightRemapYG;

	// Lidar to Stereo
	extern Mat33 RL2SG;
	extern Vec3 TL2SG;

	extern int IsRotateG;

	void SetDispG(int MaxDisp, int InvalidDisp, int sublvl);
	void InitialGlobalInfo(std::string path);
	void SetBaselineScaleG(int scale);
}
#endif