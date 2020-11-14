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



#include "stdio.h"
#include <iostream>
#include "globalCalib.h"
 
namespace SFRSS
{
	// the pyramid of Target Size
	int wG[PYR_LEVELS], hG[PYR_LEVELS];
	double fxG[PYR_LEVELS], fyG[PYR_LEVELS],
		   cxG[PYR_LEVELS], cyG[PYR_LEVELS];

	double fxiG[PYR_LEVELS], fyiG[PYR_LEVELS],
		   cxiG[PYR_LEVELS], cyiG[PYR_LEVELS];

	Mat33 KG[PYR_LEVELS], KiG[PYR_LEVELS];
	cv::Mat cvKG[PYR_LEVELS], cvKiG[PYR_LEVELS];
	cv::Mat cvKGX2;

	int rawColsG, rawRowsG;

	// Raw intrinisic, K0 and K1 for left and right respectively
	cv::Mat cvK0G, cvK0iG;
	cv::Mat cvK1G, cvK1iG;
	cv::Mat cvDistCoef0, cvDistCoef1;

	// Opencv matrixs for undist images
	cv::Mat M1lG, M2lG, M1rG, M2rG;
	cv::Mat unM1lG, unM2lG, unM1rG, unM2rG;
	cv::Mat unM1rx2G, unM2rx2G;

	// Map from undist coordinates to raw image
	float * ptrLeftRemapXG;
	float * ptrLeftRemapYG;
	float * ptrRightRemapXG;
	float * ptrRightRemapYG;

	// Extrinisic 
	double bfxG;
	double bfxG2;
	double baselineG;

	double RowTimeG = 2e-5;
	int PyrLevelsUsedG = 4;
	int baselineScaleG = 1;

	Mat33 RLRG;
	Vec3 TLRG;
	cv::Mat cvRLRG, cvTLRG;

	Mat33 RectLeftRG;

	Mat33 RL2SG;
	Vec3 TL2SG;

	// Others
	int invalidDispG, subPixelLevlG;
	int maxDisparityG[PYR_LEVELS];
	float wM3G;
	float hM3G;

	int IsRotateG = 0;

	// This function is used to determine how many levels in image pyramid.
    // And construct the intrinsic matrix for every level
	void setGlobalCalib(int w, int h, Mat33 K){
		double pixel_offset = 0.5;

		wM3G = w - 3;
		hM3G = h - 3;

		wG[0] = w;
		hG[0] = h;
		KG[0] = K;
		fxG[0] = K(0,0);
		fyG[0] = K(1,1);
		cxG[0] = K(0,2) - pixel_offset;
		cyG[0] = K(1,2) - pixel_offset;
		KiG[0] = KG[0].inverse();
		fxiG[0] = KiG[0](0,0);
		fyiG[0] = KiG[0](1,1);
		cxiG[0] = KiG[0](0,2);
		cyiG[0] = KiG[0](1,2);

		for (int level = 1; level < PyrLevelsUsedG; ++level)
		{
			wG[level] = w >> level;
			hG[level] = h >> level;

			fxG[level] = fxG[level-1] * 0.5;
			fyG[level] = fyG[level-1] * 0.5;
			cxG[level] = (cxG[0] + pixel_offset) / ((int)1<<level) - pixel_offset; 
			cyG[level] = (cyG[0] + pixel_offset) / ((int)1<<level) - pixel_offset; 

			// synthetic
			KG[level]  << fxG[level], 0.0, cxG[level], 0.0, fyG[level], cyG[level], 0.0, 0.0, 1.0;	
			KiG[level] = KG[level].inverse();

			fxiG[level] = KiG[level](0,0);
			fyiG[level] = KiG[level](1,1);
			cxiG[level] = KiG[level](0,2);
			cyiG[level] = KiG[level](1,2);
		}

		for (int level = 0; level < PyrLevelsUsedG; ++level){
			cvKG[level] = cv::Mat(3,3,CV_64F,cv::Scalar(0));
			cvKG[level].at<double>(0,0) = KG[level](0,0);
			cvKG[level].at<double>(0,1) = KG[level](0,1);
			cvKG[level].at<double>(0,2) = KG[level](0,2);

			cvKG[level].at<double>(1,0) = KG[level](1,0);
			cvKG[level].at<double>(1,1) = KG[level](1,1);
			cvKG[level].at<double>(1,2) = KG[level](1,2);

			cvKG[level].at<double>(2,0) = KG[level](2,0);
			cvKG[level].at<double>(2,1) = KG[level](2,1);
			cvKG[level].at<double>(2,2) = KG[level](2,2);


			cvKiG[level] = cv::Mat(3,3,CV_64F,cv::Scalar(0));
			cvKiG[level].at<double>(0,0) = KiG[level](0,0);
			cvKiG[level].at<double>(0,1) = KiG[level](0,1);
			cvKiG[level].at<double>(0,2) = KiG[level](0,2);

			cvKiG[level].at<double>(1,0) = KiG[level](1,0);
			cvKiG[level].at<double>(1,1) = KiG[level](1,1);
			cvKiG[level].at<double>(1,2) = KiG[level](1,2);

			cvKiG[level].at<double>(2,0) = KiG[level](2,0);
			cvKiG[level].at<double>(2,1) = KiG[level](2,1);
			cvKiG[level].at<double>(2,2) = KiG[level](2,2);
		}

		cvKGX2 = cvKG[0].clone();
		cvKGX2.at<double>(0,0) = cvKGX2.at<double>(0,0)*2;
		cvKGX2.at<double>(1,1) = cvKGX2.at<double>(1,1)*2;
		cvKGX2.at<double>(0,2) = (cvKGX2.at<double>(0,2) + pixel_offset)*2 - pixel_offset;
		cvKGX2.at<double>(1,2) = (cvKGX2.at<double>(1,2) + pixel_offset)*2 - pixel_offset;
	}

	void SetDispG(int MaxDisp, int InvalidDisp, int sublvl){
		subPixelLevlG = sublvl;
		invalidDispG = InvalidDisp;
		maxDisparityG[0] = MaxDisp;
		for (int lvl = 1; lvl < PyrLevelsUsedG; ++lvl){
			maxDisparityG[lvl] = maxDisparityG[lvl - 1] / 2;
		}
	}

	void SetBaselineScaleG(int scale){
		baselineScaleG = scale;
	}

	void InitialLidarToStereo(){
		Mat33 R;
		Vec3 T;

		R << 0.00232236,0.04739261,0.99887364,
			-0.99978071,-0.02067882,0.00330560,
			0.02081219,-0.99866227,0.04733420;
		T << -0.07477049,-0.06312950,-0.14186946;

		Vec3 T1;
		T1 << -0.05, -0.17, 0.0;
		T1 = -T1;

		Mat33 R2, R2i;
		R2 << 0.999991, 0.000471012, -0.0043038,
			-0.000456539, 0.999994, 0.00330919, 
			0.00430524, -0.00330708, 0.999986;
		Vec3 T2;
		T2 << 0.0047809, 0.0111018, -0.00224068;
		
		T2 = -R2.inverse() * T2;
		R2i = R2.inverse();

		T = T2 + R2i*(T1 + T);
		R = R2i * R;

		RL2SG = R.inverse();
		TL2SG = -R.inverse() * T;
	}

	void InitialGlobalInfo(std::string path){

		cv::FileStorage fSettings(path, cv::FileStorage::READ);
		std::cout << "Config file:" << path << std::endl;

		// Raw Width/Height
		rawColsG = (int)fSettings["Camera.width"];
		rawRowsG = (int)fSettings["Camera.height"];

		// Load intrinisic
		fSettings["M1"] >> cvK0G;
		fSettings["M2"] >> cvK1G;
		cvK0iG = cvK0G.inv();
		cvK1iG = cvK1G.inv();
		fSettings["D1"] >> cvDistCoef0;
		fSettings["D2"] >> cvDistCoef1;
		cvDistCoef0 = cvDistCoef0.t();  // (1,N) -> (N,1)
		cvDistCoef1 = cvDistCoef1.t();
		if (cvDistCoef0.at<double>(0,4) == 0.0){
			cvDistCoef0.resize(4);
		}
		else{
			cvDistCoef0.resize(5);
		}

		if (cvDistCoef1.at<double>(0,4) == 0.0){
			cvDistCoef1.resize(4);
		}
		else{
			cvDistCoef1.resize(5);
		}

		// Load extrinisc
		fSettings["R"] >> cvRLRG;
		fSettings["T"] >> cvTLRG;

		cv::Mat cvRLRt = cvRLRG.t();
		memcpy(RLRG.data(), cvRLRt.ptr<uchar>(0), sizeof(double)*9);
		memcpy(TLRG.data(), cvTLRG.ptr<uchar>(0), sizeof(double)*3);
		baselineG = (RLRG.inverse() * TLRG).norm();
		
		std::cout << -RLRG.inverse() * TLRG << std::endl; // 

		if(cvK0G.empty() || cvK1G.empty() || cvDistCoef0.empty() || cvDistCoef1.empty()){
			std::cout << "ERROR: Calibration parameters to rectify stereo are missing!" << std::endl;
			return;
		}

		RowTimeG = (double)fSettings["ScanlineTime"];

		// Construct target intrinisic
		int TargetWidth = (int)fSettings["Target.width"];
		int TargetHeight = (int)fSettings["Target.height"];
		double TargetFx = (double)fSettings["Target.fx"];
		double TargetFy = (double)fSettings["Target.fy"];
		double TargetCx = (double)fSettings["Target.cx"];
		double TargetCy = (double)fSettings["Target.cy"];

		cv::Mat cvTargetK = cv::Mat(3, 3, CV_64F, 0.0);
		cvTargetK.at<double>(0,0) = TargetFx*TargetWidth;
		cvTargetK.at<double>(0,2) = TargetCx*TargetWidth;
		cvTargetK.at<double>(1,1) = TargetFy*TargetHeight;
		cvTargetK.at<double>(1,2) = TargetCy*TargetHeight;
		cvTargetK.at<double>(2,2) = 1.0;

		bfxG = baselineG * cvTargetK.at<double>(0,0);
		bfxG2 = bfxG * 2;

		Mat33 TargetK;
		cv::Mat cvTargetKt = cvTargetK.t();
		memcpy(TargetK.data(), cvTargetKt.ptr<uchar>(0), sizeof(double)*9);

		// Update global pyramid parametors by target info 
		setGlobalCalib(TargetWidth, TargetHeight, TargetK);


		std::string model;
		fSettings["Model"] >> model;

		// build remap
		if (model.compare("RadTan") == 0){ // RadTan
			cv::Mat cvR0G, cvR1G;
			fSettings["R1"] >> cvR0G;
			fSettings["R2"] >> cvR1G;

			if(cvR0G.empty() || cvR1G.empty()){
				std::cout << "ERROR: Rectfied parameters are missing!" << std::endl;
				return;
			}

			// rotate 90 degree for new config (baseline = 25cm)
			fSettings["IsRotate"] >> IsRotateG;

			if (IsRotateG){
				cv::Mat r90(3,3,CV_64F, cv::Scalar(0));
				r90.at<double>(0,1) = -1.0;
				r90.at<double>(1,0) = 1.0;
				r90.at<double>(2,2) = 1.0;
				cvR0G = r90 * cvR0G;
				cvR1G = r90 * cvR1G;
			}

			cv::Mat cvRectLeftRG = cvR0G.t();
			memcpy(RectLeftRG.data(), cvRectLeftRG.ptr<uchar>(0), sizeof(double)*9);

			cv::initUndistortRectifyMap(cvK0G, cvDistCoef0, cvR0G, cvKG[0], cv::Size(TargetWidth,TargetHeight), CV_32F, M1lG, M2lG);
			cv::initUndistortRectifyMap(cvK1G, cvDistCoef1, cvR1G, cvKG[0], cv::Size(TargetWidth,TargetHeight), CV_32F, M1rG,M2rG);

			cv::Mat reye = cv::Mat::eye(3,3,CV_64F);
			std::cout << reye << std::endl;
			cv::initUndistortRectifyMap(cvK0G, cvDistCoef0, reye, cvKG[0], cv::Size(TargetWidth,TargetHeight), CV_32F, unM1lG, unM2lG);
			cv::initUndistortRectifyMap(cvK1G, cvDistCoef1, reye, cvKG[0], cv::Size(TargetWidth,TargetHeight), CV_32F, unM1rG, unM2rG);
			// For better cost volume
			cv::initUndistortRectifyMap(cvK1G, cvDistCoef1, reye, cvKGX2, cv::Size(TargetWidth*2,TargetHeight*2), CV_32F, unM1rx2G, unM2rx2G);
		}
		else{
			// "Equ model"
			cv::Mat cvR0G, cvR1G;
			cv::Mat P1, P2, Q;
			// cv::Rect validRoi[2];

			cv::fisheye::stereoRectify(cvK0G, cvDistCoef0,
						cvK1G, cvDistCoef1,
						cv::Size(rawColsG, rawRowsG), cvRLRG, cvTLRG, cvR0G, cvR1G, P1, P2, Q,
						cv::CALIB_ZERO_DISPARITY, cv::Size(TargetWidth, TargetHeight));


			// rotate 90 degree for new config (baseline = 25cm)
			// fSettings["IsRotate"] >> IsRotateG;

			// if (IsRotateG){
			// 	cv::Mat r90(3,3,CV_64F, cv::Scalar(0));
			// 	r90.at<double>(0,1) = -1.0;
			// 	r90.at<double>(1,0) = 1.0;
			// 	r90.at<double>(2,2) = 1.0;
			// 	cvR0G = r90 * cvR0G;
			// 	cvR1G = r90 * cvR1G;
			// }
			cv::fisheye::initUndistortRectifyMap(cvK0G, cvDistCoef0, cvR0G, cvKG[0], cv::Size(TargetWidth,TargetHeight), CV_32F, M1lG, M2lG);
			cv::fisheye::initUndistortRectifyMap(cvK1G, cvDistCoef1, cvR1G, cvKG[0], cv::Size(TargetWidth,TargetHeight), CV_32F, M1rG,M2rG);

			cv::Mat reye = cv::Mat::eye(3,3,CV_64F);
			std::cout << reye << std::endl;
			cv::fisheye::initUndistortRectifyMap(cvK0G, cvDistCoef0, reye, cvKG[0], cv::Size(TargetWidth,TargetHeight), CV_32F, unM1lG, unM2lG);
			cv::fisheye::initUndistortRectifyMap(cvK1G, cvDistCoef1, reye, cvKG[0], cv::Size(TargetWidth,TargetHeight), CV_32F, unM1rG, unM2rG);
			// For better cost volume
			cv::fisheye::initUndistortRectifyMap(cvK1G, cvDistCoef1, reye, cvKGX2, cv::Size(TargetWidth*2,TargetHeight*2), CV_32F, unM1rx2G, unM2rx2G);
		}
		
		ptrLeftRemapXG = unM1lG.ptr<float>(0);
		ptrLeftRemapYG = unM2lG.ptr<float>(0);
		ptrRightRemapXG = unM1rG.ptr<float>(0);
		ptrRightRemapYG = unM2rG.ptr<float>(0);

		// Print all parametors
		std::cout << std::endl << "Camera Parameters: " << std::endl;
		std::cout << "- fx0: " << cvK0G.at<double>(0,0) << std::endl;
		std::cout << "- fy0: " << cvK0G.at<double>(1,1) << std::endl;
		std::cout << "- cx0: " << cvK0G.at<double>(0,2) << std::endl;
		std::cout << "- cy0: " << cvK0G.at<double>(1,2) << std::endl;
		std::cout << "- k1: " << cvDistCoef0.at<double>(0) << std::endl;
		std::cout << "- k2: " << cvDistCoef0.at<double>(1) << std::endl;
		if(cvDistCoef0.rows==5)
			std::cout << "- k3: " << cvDistCoef0.at<double>(4) << std::endl;
		std::cout << "- p1: " << cvDistCoef0.at<double>(2) << std::endl;
		std::cout << "- p2: " << cvDistCoef0.at<double>(3) << std::endl;
		std::cout << std::endl;

		std::cout << "- fx1: " << cvK1G.at<double>(0,0) << std::endl;
		std::cout << "- fy1: " << cvK1G.at<double>(1,1) << std::endl;
		std::cout << "- cx1: " << cvK1G.at<double>(0,2) << std::endl;
		std::cout << "- cy1: " << cvK1G.at<double>(1,2) << std::endl;
		std::cout << "- k1: " << cvDistCoef1.at<double>(0) << std::endl;
		std::cout << "- k2: " << cvDistCoef1.at<double>(1) << std::endl;
		if(cvDistCoef1.rows==5)
			std::cout << "- k3: " << cvDistCoef1.at<double>(4) << std::endl;
		std::cout << "- p1: " << cvDistCoef1.at<double>(2) << std::endl;
		std::cout << "- p2: " << cvDistCoef1.at<double>(3) << std::endl;
		std::cout << std::endl;

		std::cout << "- TargetWidth" << ": " << TargetWidth << std::endl;
		std::cout << "- TargetHeight" << ": " << TargetHeight << std::endl;
		std::cout << "- TargetFx" << ": " << TargetFx << std::endl;
		std::cout << "- TargetFy" << ": " << TargetFy << std::endl;
		std::cout << "- TargetCx" << ": " << TargetCx << std::endl;
		std::cout << "- TargetCy" << ": " << TargetCy << std::endl;

		InitialLidarToStereo();
	}
}
