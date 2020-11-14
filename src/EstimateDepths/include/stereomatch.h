#pragma once

#include <iomanip>
#include <string>
#include <chrono>
#include <memory>
#include "libsgm.h"
#include <vector>
#include <cuda_runtime.h>
#include "../../System/System.h"
#include "../../util/NumType.h"
#include "../../util/FramePym.h"
#include <string>

namespace SFRSS{

struct device_buffer
{
	device_buffer() : data(nullptr) {}
	device_buffer(size_t count) { allocate(count); }
	void allocate(size_t count) { cudaMalloc(&data, count); }
	~device_buffer() { cudaFree(data); }
	void* data;
};

class System;
class StereoMatch{
public:
    StereoMatch(System* pSys, std::string strSettingPath);
    ~StereoMatch();
    int InitStereoMatch(int diparity_size, int input_depth, int output_depth, bool IsSubPixel);
    int InitRSProjection(int GridSize, int Border);
 
    int SetImagePair(FramePym * leftFrame, FramePym * rightFrame);

    double StereoMatchFinalMode4(int lvl, void * ptr_result, bool IsInterpolate = false);

    int RSProjection3(std::vector<double> motionState, int lvl);

    int CheckCorrespondence(int testLvl, bool left2right = true);
    int CheckLRDisparity(uint16_t * leftRaw, uint16_t * rightRaw, int lvl);

    void ShowDisparity(void * ptr_disparity, std::string win_name, int lvl, bool flip = false);
    void SetShowMaxDisparity(int max);

    void GenerateBaselineMap(cv::Mat & baselineMap);

private:
    sgm::StereoSGM * ptr_sgm;
    System * mpSystem;

    int ShowMaxDisparity; // for view

    // SGM
    int m_disparity_size;
    int m_sgm_input_depth;
    int m_sgm_output_depth;
    int m_invalid_disp;
    int m_input_bytes;
    int m_output_bytes;

    uint8_t * m_ptrLeftMask;
    uint8_t * m_ptrRightMask;

    unsigned int * ptrLeftCensus[PYR_LEVELS];
    unsigned int * ptrRightCensus[PYR_LEVELS];
    unsigned int * ptrRightCensusRaw;
    unsigned int * ptrCostVolumes;

    // cv::Mat * ptr_disparity_mat;
    device_buffer * ptr_d_left;
    device_buffer * ptr_d_right;
    device_buffer * ptr_d_disparity;
    device_buffer * ptr_d_cost;

    int m_pattern_size;
    uint8_t * ptrFinalCost;

    FramePym * mleftFrame;
    FramePym * mrightFrame;

};

}