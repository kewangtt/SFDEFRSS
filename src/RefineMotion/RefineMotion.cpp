#include <stdio.h>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/version.hpp>
#include <cmath>
#include <Eigen/Dense>

#include "CoarseInitializer.h"
#include "RefineMotion.h"
#include "../util/FramePym.h"
#include "../util/globalFuncs.h"
#include "../util/globalCalib.h"

using namespace std;
using namespace SFRSS;
using namespace cv;
using namespace Eigen;

namespace SFRSS
{

int BuildInitialPoints3(uint16_t * disparity, Pnt** points_out, int * points_num, int lvl) 
{
    Pnt* local_points;
    // bool IsSkip = false;
    
    local_points = points_out[lvl];
    
    int sucess = 0;
    double curbfx = fxG[lvl] * baselineG * baselineScaleG;
    for(int ii = 0; ii < points_num[lvl]; ++ii)
    {
        local_points[ii].isGood = true;
        local_points[ii].valid_idepth_num = 0;

        int iid = int(local_points[ii].u + local_points[ii].v*wG[lvl] + 0.0001);

        // Select disparity candidates
        if (disparity[iid] == invalidDispG){
            local_points[ii].isGood = false;
            continue;
        }    

        float disp = disparity[iid];
        disp = disp / subPixelLevlG + 1;

        local_points[ii].idepth_initial = local_points[ii].idepth_new = local_points[ii].idepth = local_points[ii].ori_idepth = local_points[ii].idepth_candidates[0] = disp / curbfx;
        local_points[ii].valid_idepth_num = 1;
        sucess += 1;
    }

    printf("Lvl %d: %d/%d\n",lvl, sucess, points_num[lvl]);

    return 0;
}

RefineMotion::RefineMotion(System* pSys, double rowtime): mpSystem(pSys){

    mpCoarseInitializer = new CoarseInitializer(wG[0], hG[0]);
}

RefineMotion::~RefineMotion(){
    if (mpCoarseInitializer){
        delete mpCoarseInitializer;
    }
}

// Transfer depth from world cooridinate to left local cooridinate
int TransferDepth(Pnt ** points, int * numPoints, int lvl, vector<double> motionState, double mRowTime){

	Vec3 Velocity(motionState[3] * mRowTime, motionState[4] * mRowTime, motionState[5] * mRowTime);
	Vec3 Angular(motionState[0] * mRowTime, motionState[1] * mRowTime, motionState[2] * mRowTime);

	double AngSpeed = Angular.norm();
	Vec3 AngAxis;
	if (AngSpeed < 0.0000000001){
	    AngAxis[0] = 1.0;
	    AngAxis[1] = 1.0;
	    AngAxis[2] = 1.0;
		AngAxis = AngAxis / AngAxis.norm();
	}
	else{
		AngAxis = Angular / AngSpeed;
	}

    Pnt * localPoints = points[lvl];

    int scale = 1 << lvl;
    Vec3 curLeftPoint;

    for(int ii = 0; ii < numPoints[lvl]; ++ii)
    {
        if (!localPoints[ii].isGood){
            continue;
        }

        curLeftPoint << localPoints[ii].u, localPoints[ii].v, 1.0;
        curLeftPoint = KiG[lvl] * curLeftPoint;

        float rawU = (localPoints[ii].u + 0.5)*scale - 0.5;
        float rawV = (localPoints[ii].v + 0.5)*scale - 0.5;
        // int liid = rawV*wG[0] + rawU;
        double leftRow = getInterpolatedElement(ptrLeftRemapYG, rawU, rawV, wG[lvl]);

        // construct ray
        Mat33 Rl = Eigen::AngleAxisd(AngSpeed*leftRow, AngAxis).matrix();
        Mat33 lrinv = Rl.inverse();
        Vec3 start = -leftRow*lrinv*Velocity;
        Vec3 inc = lrinv * curLeftPoint;

        double depthWorld = 1 / localPoints[ii].ori_idepth;

        // printf("old:%f\n", localPoints[ii].ori_idepth);

        double lamba = (depthWorld - start[2]) / inc[2];
        curLeftPoint = start + lamba * inc;

        // now, curLeftPoint is in world coordinate
        // transfer curLeftPoint to the left point coordinate
        curLeftPoint = Rl * curLeftPoint + leftRow*Velocity;

        // idepth in local point coordinate
        localPoints[ii].idepth_initial = localPoints[ii].idepth_new  = localPoints[ii].idepth = localPoints[ii].ori_idepth = 1 / curLeftPoint[2];

        // printf("new:%f\n", localPoints[ii].idepth_new);
    }
    return 0;
}

void RefineMotion::InitRefineMotion(FramePym * ptrLeft, FramePym * ptrRight){
    // Build key points in left FramePym
    mpCoarseInitializer->setLeftFramePym(ptrLeft);
    mpCoarseInitializer->setRightFramePym(ptrRight);

}

// disparity: 4 buffer, there are 4 slots for every pixels
vector<double> RefineMotion::Refine2(uint16_t * disparity, vector<double> motionState, int pyramidLvl, bool FixTranslation){

    // BuildInitialPoints3 for disparity map with 4 slots
    BuildInitialPoints3(disparity, mpCoarseInitializer->points, mpCoarseInitializer->numPoints, pyramidLvl);

    // Transfer depth from world coordinate to local coordinate
    TransferDepth(mpCoarseInitializer->points, mpCoarseInitializer->numPoints, pyramidLvl, motionState, RowTimeG);

    vector<double> finalState = mpCoarseInitializer->Optimize3_withdepths_MT(motionState, pyramidLvl, FixTranslation); // _MT

    return finalState;
}


}