#include <stdio.h>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/version.hpp>
#include <cmath>
#include <vector>

#include "../System/System.h"
#include "CoarseInitializer.h"
#include "../util/FramePym.h"

// #include "util/globalCalib.h"
// #include "HessianBlocks.h"
// #include "PixelSelector2.h"
// #include "CoarseInitializer.h"


using namespace std;
// using namespace SFRSS;
// using namespace cv;
// using namespace Eigen;

#ifndef REFINEMOTION_H
#define REFINEMOTION_H

namespace SFRSS
{
class System;
class CoarseInitializer;
class RefineMotion{
public:
    RefineMotion(System* pSys, double rowtime);
    ~RefineMotion();
    void InitRefineMotion(FramePym * ptrLeft, FramePym * ptrRight);
    vector<double> Refine(float * disparity, vector<double> motionState, int pyramidLvl, bool IsFixTranslation = false);
    vector<double> Refine2(uint16_t * disparity, vector<double> motionState, int pyramidLvl, bool IsFixTranslation = false);
    void EvalConvergeAbility(int lvl);
private:
    System * mpSystem;
    CoarseInitializer * mpCoarseInitializer;
};
}
#endif