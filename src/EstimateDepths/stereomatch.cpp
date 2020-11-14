#include <opencv2/opencv.hpp>
#include <Eigen/Geometry> 
#include "./include/stereomatch.h"
#include "./include/libsgm.h"
#include "../util/globalCalib.h"
#include "../util/FramePym.h"

namespace SFRSS{
using namespace cv;
using namespace Eigen;

#define PATTERN_SIZE 30

// PreCalibrated relative pose

int mGridSize[PYR_LEVELS];
int mBorder[PYR_LEVELS];

int mAnchorHeight[PYR_LEVELS];
int mAnchorWidth[PYR_LEVELS];
int * mptrAnchorU[PYR_LEVELS], * mptrAnchorV[PYR_LEVELS];
double * mptrAncherNormalzied[PYR_LEVELS];
double * mptrAnchorLeftRawCoordi[PYR_LEVELS];
double *mptrInitialRightRawCoordi[PYR_LEVELS];
double *mptrInitialRightTargetCoordi[PYR_LEVELS];

double invDepths[PYR_LEVELS][1024];
uint8_t * mDisparityMask[PYR_LEVELS];
uint8_t * Right2LeftMask[PYR_LEVELS];

float * ptrInterpolateCoeffsTL[PYR_LEVELS];
float * ptrInterpolateCoeffsTR[PYR_LEVELS];
float * ptrInterpolateCoeffsBL[PYR_LEVELS];
float * ptrInterpolateCoeffsBR[PYR_LEVELS];

double * mptrBaselineAcc;
double * mptrBaselineNum;
double * mptrBaselineAve;
uint8_t * mptrdispscale;
double * mptrEpipolarLength;

float * ptrResCoordinate;
float * ptrResCoordinateR2L;
char * ptrResOffset;

float * ptrCoordiBlocks;
char * ptrOffsetBlocks;

float * ptrTestInitialRightRow;
float * ptrTestFarLeftRow;
float * ptrTestCloseLeftRow;
uint8_t * mptrFinalCost;

uint16_t * mptrRightDisparity2;
uint16_t * mptrAggregationCost;

uint32_t * ptrCostVolumes2 = NULL;

#define _DEFAULT_ERROR_COORDI (-1)
#define _DEFAULT_ERROR_OFFSET (-64)

// (9*9)
int circle_pattern[PATTERN_SIZE][4] = {
	{0,-1,0,1},
	{-1,-1,1,1},
	{-1,0,1,0},
	{-1,1,1,-1},
	{0,-2,0,2},
	{-1,-2,1,2},
	{-2,-1,2,1},
	{-2,0,2,0},
	{-2,1,2,-1},
	{-1,2,1,-2},
	{0,-3,0,3},
	{-1,-3,1,3},
	{-2,-2,2,2},
	{-3,-1,3,1},
	{-3,0,3,0},
	{-3,1,3,-1},
	{-2,2,2,-2},
	{-1,3,1,-3},
	{0,-4,0,4},
	{-1,-4,1,4},
	{-2,-3,2,3},
	{-3,-3,3,3},
	{-3,-2,3,2},
	{-4,-1,4,1},
	{-4,0,4,0},
	{-4,1,4,-1},
	{-3,2,3,-2},
	{-3,3,3,-3},
	{-2,3,2,-3},
	{-1,4,1,-4}
};


// 7*9
int CensusTransformSquare(uint32_t * dest, uint8_t * img, int img_h, int img_w, int win_h, int win_w){

	memset(dest, 0, sizeof(uint32_t)*img_h*img_w);
	int half_kw = win_w / 2;
	int half_kh = win_h / 2;
	uint32_t f = 0;
	for(int y = half_kh; y < img_h - half_kh; ++y){
		for(int x = half_kw; x < img_w - half_kw; ++x){
			f = 0;
			// 3 * 9
			for(int dy = -half_kh; dy < 0; ++dy){
				const int smem_y1 = y + dy;
				const int smem_y2 = y - dy;
				for(int dx = -half_kw; dx <= half_kw; ++dx){
					const int smem_x1 = x + dx;
					const int smem_x2 = x - dx;
					unsigned char a = img[smem_y1*img_w + smem_x1];
					unsigned char b = img[smem_y2*img_w +smem_x2];
					f = (f << 1) | (a > b);
				}
			}

			for(int dx = -half_kw; dx < 0; ++dx){
				const int smem_x1 = x + dx;
				const int smem_x2 = x - dx;
				unsigned char a = img[y*img_w + smem_x1];
				unsigned char b = img[y*img_w + smem_x2];
				f = (f << 1) | (a > b);
			}
			dest[y*img_w + x] = f;
		}
	}

	return 0;
}

template<typename T>
int CensusTransformCircle(uint32_t * dest, T * img, int img_h, int img_w, int radius, uint8_t* mask){

	memset(dest, 0, sizeof(uint32_t)*img_h*img_w);
	int half_kw = radius / 2;
	int half_kh = radius / 2;
	uint32_t f = 0;
	for(int y = half_kh; y < img_h - half_kh; ++y){
		for(int x = half_kw; x < img_w - half_kw; ++x){
			f = 0;
			if (mask && mask[y*img_w + x] == 0){ // mask  = Null, in final mode
				continue;
			}
			for(int kk = 0; kk < PATTERN_SIZE; ++kk){
				const int smem_y1 = y + circle_pattern[kk][0];
				const int smem_y2 = y + circle_pattern[kk][2];
				const int smem_x1 = x + circle_pattern[kk][1];
				const int smem_x2 = x + circle_pattern[kk][3];
				unsigned char a = img[smem_y1*img_w + smem_x1];
				unsigned char b = img[smem_y2*img_w + smem_x2];
				f = (f << 1) | (a > b);
			}
			dest[y*img_w + x] = f;
		}
	}

	return 0;
}



StereoMatch::StereoMatch(System* pSys, std::string strSettingPath): mpSystem(pSys){
	ptr_d_left = NULL;
	ptr_d_right = NULL;
	ptr_d_disparity = NULL;
	ptr_sgm = NULL;

	ptr_d_cost = NULL;
	ptrCostVolumes = NULL;

	ptrRightCensusRaw = NULL;

	for (int lvl = 0; lvl < PyrLevelsUsedG; ++lvl){
		ptrLeftCensus[lvl] = NULL;
		ptrRightCensus[lvl] = NULL;

		mptrAnchorU[lvl] = NULL;
		mptrAnchorV[lvl] = NULL;
		mptrAncherNormalzied[lvl] = NULL;
		mptrAnchorLeftRawCoordi[lvl] = NULL;
		mptrInitialRightRawCoordi[lvl] = NULL;
		mptrInitialRightTargetCoordi[lvl] = NULL;

		ptrInterpolateCoeffsTL[lvl] = NULL;
		ptrInterpolateCoeffsTR[lvl] = NULL;
		ptrInterpolateCoeffsBL[lvl] = NULL;
		ptrInterpolateCoeffsBR[lvl] = NULL;

		mDisparityMask[lvl] = (uint8_t *)malloc(wG[lvl] * hG[lvl]);
		memset(mDisparityMask[lvl], 1, wG[lvl] * hG[lvl]);

		Right2LeftMask[lvl] = (uint8_t *)malloc(wG[lvl] * hG[lvl]);
		memset(Right2LeftMask[lvl], 1, wG[lvl] * hG[lvl]);

		float localScale = 1 << lvl;
		for (int ii = 0; ii < wG[lvl]*hG[lvl]; ++ii){
			float curv = int(ii / wG[lvl]);
			float curu = ii % wG[lvl];
			int curvi = int(((curv + 0.5) * localScale) - 0.5 + 0.5);
			int curui = int(((curu + 0.5) * localScale) - 0.5 + 0.5);
			int iid = curvi*wG[0] + curui;
			if (!(ptrLeftRemapXG[iid] > 0 && 
				ptrLeftRemapXG[iid]< rawColsG - 1 && 
				ptrLeftRemapYG[iid]> 0 && 
				ptrLeftRemapYG[iid] < rawRowsG-1)){
				mDisparityMask[lvl][ii] = 0;
			}

			if (!(ptrRightRemapXG[iid] > 0 && 
				ptrRightRemapXG[iid] < rawColsG - 1 && 
				ptrRightRemapYG[iid] > 0 && 
				ptrRightRemapYG[iid] < rawRowsG-1)){
				Right2LeftMask[lvl][ii] = 0;
			}
		}
	}

	ptrResCoordinate = NULL;
	ptrResCoordinateR2L = NULL;
	ptrResOffset = NULL;

	ptrCoordiBlocks = NULL;
	ptrOffsetBlocks = NULL;

	mptrBaselineAcc = NULL;
	mptrBaselineNum = NULL;

	mptrBaselineAve = NULL;
	mptrdispscale = NULL;

	mptrEpipolarLength = NULL;


	ptrTestInitialRightRow = NULL;
	ptrTestFarLeftRow = NULL;
	ptrTestCloseLeftRow = NULL;

	mptrRightDisparity2 = NULL;
	mptrAggregationCost = NULL;

	mptrFinalCost = NULL;
	ptrFinalCost = NULL;

}


StereoMatch::~StereoMatch(){
	if (ptr_sgm) delete ptr_sgm;
	if (ptr_d_left) delete ptr_d_left;
	if (ptr_d_right) delete ptr_d_right;
	if (ptr_d_disparity) delete ptr_d_disparity;
	if (ptr_d_cost) delete ptr_d_cost;


	if (ptrRightCensusRaw) delete ptrRightCensusRaw;
	if (ptrCostVolumes) delete ptrCostVolumes;

	if (ptrResCoordinate) free(ptrResCoordinate);
	if (ptrResCoordinateR2L) free(ptrResCoordinateR2L);
	if (ptrResOffset) free(ptrResOffset);

	if (ptrCoordiBlocks) free(ptrCoordiBlocks);
	if (ptrOffsetBlocks) free(ptrOffsetBlocks);
	if (ptrFinalCost) free(ptrFinalCost);
	if (mptrFinalCost) free(mptrFinalCost);

	if (ptrCostVolumes2) free(ptrCostVolumes2);

	for (int lvl = 0; lvl < PyrLevelsUsedG; ++lvl){
		if (ptrLeftCensus[lvl]) delete ptrLeftCensus[lvl];
		if (ptrRightCensus[lvl]) delete ptrRightCensus[lvl];

		if (mptrAnchorU[lvl]) free(mptrAnchorU[lvl]);
		if (mptrAnchorV[lvl]) free(mptrAnchorV[lvl]);
		if (mptrAncherNormalzied[lvl]) free(mptrAncherNormalzied[lvl]);
		if (mptrAnchorLeftRawCoordi[lvl]) free(mptrAnchorLeftRawCoordi[lvl]);
		if (mptrInitialRightRawCoordi[lvl]) free(mptrInitialRightRawCoordi[lvl]);
		if (mptrInitialRightTargetCoordi[lvl]) free(mptrInitialRightTargetCoordi[lvl]);

		if (ptrInterpolateCoeffsTL[lvl]) free(ptrInterpolateCoeffsTL[lvl]);
		if (ptrInterpolateCoeffsTR[lvl]) free(ptrInterpolateCoeffsTR[lvl]);
		if (ptrInterpolateCoeffsBL[lvl]) free(ptrInterpolateCoeffsBL[lvl]);
		if (ptrInterpolateCoeffsBR[lvl]) free(ptrInterpolateCoeffsBR[lvl]);

		if (mDisparityMask[lvl]) free(mDisparityMask[lvl]);
		if (Right2LeftMask[lvl]) free(Right2LeftMask[lvl]);
	}
}



int CensusTransformCircle_Test(uint32_t * dest, uint8_t * img, int img_h, int img_w, int radius){

	int half_kw = radius / 2;
	int half_kh = radius / 2;
	uint32_t f = 0;
	uint32_t invalid = 1 << 31;
	for(int y = half_kh; y < img_h - half_kh; ++y){
		for(int x = half_kw; x < img_w - half_kw; ++x){
			f = 0;
			if (mptrdispscale[y*img_w + x] == 0){ // mask  = Null, in final mode
				continue;
			}
			if (img[(y - half_kh)*img_w + x] == 0 ||
				img[(y + half_kh)*img_w + x] == 0 ||
				img[y*img_w + x + half_kw] == 0 ||
				img[y*img_w + x - half_kw] == 0){
				dest[y*img_w + x] = invalid;
			}
			for(int kk = 0; kk < PATTERN_SIZE; ++kk){
				const int smem_y1 = y + circle_pattern[kk][0];
				const int smem_y2 = y + circle_pattern[kk][2];
				const int smem_x1 = x + circle_pattern[kk][1];
				const int smem_x2 = x + circle_pattern[kk][3];
				unsigned char a = img[smem_y1*img_w + smem_x1];
				unsigned char b = img[smem_y2*img_w + smem_x2];
				f = (f << 1) | (a > b);
			}
			dest[y*img_w + x] = f;
		}
	}

	return 0;
}


void CalcRightDisparity2_new(int start, int end, uint16_t * aggregationCost, uint16_t * ptrRightDisparity, int invalid, int subPixelLevlG, int lvl){


	// calc right disparity by naive method
	for (int tid = start; tid < end; ++tid){
		int v = tid / wG[lvl];
		int u = tid % wG[lvl];

		if (ptrResCoordinateR2L[(v*wG[lvl] + u)*2] < 0){
			ptrRightDisparity[v*wG[lvl] + u] = invalid;
			continue;
		}

		uint16_t localcost[1024] = {0};
		int min = 1e8; 
		int index = -1;
		for (int id = 0; id < maxDisparityG[lvl]; ++id){
			float uu = ptrResCoordinateR2L[(id*wG[lvl]*hG[lvl] + v*wG[lvl] + u)*2];
			float vv = ptrResCoordinateR2L[(id*wG[lvl]*hG[lvl] + v*wG[lvl] + u)*2 + 1];

			int left_uui = int(uu + 0.5);
			int left_vvi = int(vv + 0.5);

			localcost[id] = aggregationCost[left_vvi*wG[lvl]*maxDisparityG[0] + left_uui*maxDisparityG[0] + id];

			if (min > localcost[id]){
				min = localcost[id];
				index = id;
			}
		}
		if (index >= 0){
			ptrRightDisparity[v*wG[lvl]+u] = index * subPixelLevlG;
		}
		else{
			ptrRightDisparity[v*wG[lvl] + u] = invalid;
		}	
	}

}



void WinnerTakeAll_new2(int start, int end, uint8_t * ptrFinalCost, int channels, uint16_t * ptr_result2, uint16_t * aggregationCost, int subPixelLvl, float confidence, int invalid, int lvl){

	int width = wG[lvl];
	int height = hG[lvl];

	// Aggregation and find minimal cost
	int localcost[1024] = {0};
	for (int tid = start; tid < end; ++tid){
		int v = tid / width;
		int u = tid % width;
		int min = 1e8; 
		int index = -1;
		for (int id = 0; id < maxDisparityG[lvl]; ++id){
			int sum = 0;
			for (int ic = 0; ic < channels; ++ic){
				sum += *(ptrFinalCost + ic*width*height*maxDisparityG[0] + v*width*maxDisparityG[0] + u*maxDisparityG[0] + id);
			}
			if (min > sum){
				min = sum;
				index = id;
			}
			localcost[id] = sum;
			aggregationCost[v*width*maxDisparityG[0] + u*maxDisparityG[0] + id] = sum;
		}

		bool uniq = true;
		int scale = mptrdispscale[v*width + u];
		if (scale == 0){
			continue;
		}

		double ratio = mptrBaselineAve[v*width + u] > 2.0 ? 2.0 : mptrBaselineAve[v*width + u];
		int radius = (int)(1 / ratio * 2 + 0.5);
		for (int id = 0; id < maxDisparityG[lvl]; ++id){
			int x = localcost[id];
			// uniqueness = 0.95
			const bool uniq1 = x * confidence >= min;
			const bool uniq2 = abs(id - index) <= radius;
			uniq &= uniq1 || uniq2;
		}

		if (uniq){
			if (subPixelLvl > 1 && index != 0 && index != maxDisparityG[lvl] - 1){
				const int left = localcost[index - 1];
				const int right = localcost[index + 1];
				const int numer = left - right;
				const int denom = left - 2 * min + right;
				*(ptr_result2 + v*width + u) = index*subPixelLvl + ((numer << sgm::StereoSGM::SUBPIXEL_SHIFT) + denom) / (2 * denom);				
			}
			else{
				*(ptr_result2 + v*width + u) = index*subPixelLvl;
			}
		}
		else{
			*(ptr_result2 + v*width + u) = invalid;
		}
	}
}


void LRCheck2_new(int start, int end, uint16_t * left, uint16_t * right, int lvl){
	// LR Check
	for (int iid = start; iid < end; ++iid){
		if (left[iid] == invalidDispG){
			continue;
		}
		uint16_t d = left[iid] >> 4; // / subPixelLevlG
		float uu = ptrResCoordinate[(maxDisparityG[lvl]*iid + d)*2];
		float vv = ptrResCoordinate[(maxDisparityG[lvl]*iid + d)*2 + 1];

		if (uu < 0){
			left[iid] = invalidDispG;
			continue;
		}

		int uui = int(uu + 0.5);
		int vvi = int(vv + 0.5);

		uint16_t newd = right[vvi*wG[lvl] + uui]; // / subPixelLevlG;
		if (newd == invalidDispG){
			left[iid] = invalidDispG;
			continue;
		}

		newd = newd >> 4;

		if (fabs(float(newd) - float(d)) > 1){
			left[iid] = invalidDispG;
		}
	}
}





double StereoMatch::StereoMatchFinalMode4(int lvl, void * ptr_result, bool IsInterpolate){

    if (!ptr_sgm) {
        printf("ptr_sgm error!\n");
        return 0;
    }
	
	int localw = wG[lvl];
	int localh = hG[lvl];


	// 3. Involve Smooth term and run post-process
	cudaMemcpy(ptr_d_cost->data, ptrCostVolumes2, 4*localh*localw*maxDisparityG[0], cudaMemcpyHostToDevice);

	// new test
	// 8 channels
	ptr_sgm->InvolveSmooth2((unsigned int*)(ptr_d_cost->data), localw, localh, mptrFinalCost);

    int multiThread = true;
    int threadCnt = THREAD_CNT;
    int taskCnt = localw*localh;
    int thread_step = int(ceil(taskCnt/threadCnt));
    if(multiThread){
        std::vector<std::thread> thread_pool;
        for (int it = 0; it < threadCnt; ++it){
            int pt_start = thread_step * it;
            int pt_end = thread_step * (it + 1);

			if (pt_end > taskCnt){
				pt_end = taskCnt;
			}

            std::thread this_thread(WinnerTakeAll_new2, pt_start, pt_end, mptrFinalCost, 8, (uint16_t *)ptr_result, mptrAggregationCost, subPixelLevlG, 0.95, m_invalid_disp, lvl);
            thread_pool.push_back(std::move(this_thread));
        }
        for (unsigned int it = 0; it < thread_pool.size(); ++it){
            if(thread_pool[it].joinable())
                thread_pool[it].join();
        }
    }
    else{
        WinnerTakeAll_new2(0, taskCnt, mptrFinalCost, 8, (uint16_t *)ptr_result, mptrAggregationCost, subPixelLevlG, 0.95, m_invalid_disp, lvl);
    }
	
	multiThread = true;
    threadCnt = THREAD_CNT;
    taskCnt = localw*localh;
    thread_step = int(ceil(taskCnt/threadCnt));
    if(multiThread){
        std::vector<std::thread> thread_pool;
        for (int it = 0; it < threadCnt; ++it){
            int pt_start = thread_step * it;
            int pt_end = thread_step * (it + 1);

			if (pt_end > taskCnt){
				pt_end = taskCnt;
			}

            std::thread this_thread(CalcRightDisparity2_new, pt_start, pt_end, mptrAggregationCost, mptrRightDisparity2, m_invalid_disp, subPixelLevlG, lvl);
            thread_pool.push_back(std::move(this_thread));
        }
        for (unsigned int it = 0; it < thread_pool.size(); ++it){
            if(thread_pool[it].joinable())
                thread_pool[it].join();
        }
    }
    else{
        CalcRightDisparity2_new(0, taskCnt, mptrAggregationCost, mptrRightDisparity2, m_invalid_disp, subPixelLevlG, lvl);
    }

	// median filter
	cv::Mat disp2(localh, localw, CV_16U, ptr_result);
	medianBlur(disp2, disp2, 3);
	memcpy(ptr_result, disp2.data, localw*localh*2);
	cv::Mat disp3(hG[lvl], localw, CV_16U, mptrRightDisparity2);
	medianBlur(disp3, disp3, 3);
	memcpy(mptrRightDisparity2, disp3.data, localw*localh*2);


	for (int v = 0; v < localh; ++v){
		for (int u = 0; u < localw; ++u){
			if (mptrdispscale[v*localw + u] == 0){ 
				((uint16_t *)ptr_result)[v * localw + u] = m_invalid_disp;
			}
			if (Right2LeftMask[lvl][v*localw + u] == 0){
				((uint16_t *)mptrRightDisparity2)[v * localw + u] = m_invalid_disp;
			}
		}
	}

	multiThread = true;
    threadCnt = THREAD_CNT;
    taskCnt = localw*localh;
    thread_step = int(ceil(taskCnt/threadCnt));
    if(multiThread){
        std::vector<std::thread> thread_pool;
        for (int it = 0; it < threadCnt; ++it){
            int pt_start = thread_step * it;
            int pt_end = thread_step * (it + 1);

			if (pt_end > taskCnt){
				pt_end = taskCnt;
			}

            std::thread this_thread(LRCheck2_new, pt_start, pt_end, (uint16_t *)ptr_result, mptrRightDisparity2, lvl);
            thread_pool.push_back(std::move(this_thread));
        }
        for (unsigned int it = 0; it < thread_pool.size(); ++it){
            if(thread_pool[it].joinable())
                thread_pool[it].join();
        }
    }
    else{
        LRCheck2_new(0, taskCnt, (uint16_t *)ptr_result, mptrRightDisparity2, lvl);
    }

	int Sum = 0, Valid = 0;
	float hist[256] = {0};
	int offset = 0;
	float fsum = 0.00001;

	if (!IsInterpolate){
		for (int v = 0; v < localh; ++v){
			for (int u = 0; u < localw; ++u){
				if (mptrdispscale[v*localw + u] == 0){ 
					((uint16_t *)ptr_result)[v * localw + u] = m_invalid_disp;
				}
				else{
					Sum += 1;
					uint16_t value = ((uint16_t *)ptr_result)[v * localw + u];
					if (value >= 0 && value < maxDisparityG[lvl] * subPixelLevlG){
						Valid += 1;
					}
					else{
						((uint16_t *)ptr_result)[v * localw + u] = m_invalid_disp;
					}
				}
			}
		}
	}
	else{
		if (subPixelLevlG > 0) offset = 4;
		for (int v = 0; v < localh; ++v){
			for (int u = 0; u < localw; ++u){
				if (mptrdispscale[v*localw + u] == 0){ 
					((uint16_t *)ptr_result)[v * localw + u] = m_invalid_disp;
				}
				else{
					Sum += 1;
					uint16_t value = ((uint16_t *)ptr_result)[v * localw + u];
					if (value >= 0 && value < maxDisparityG[lvl] * subPixelLevlG){
						Valid += 1;
						int idx = value >> offset;
						hist[idx] += 1;
						fsum += 1;
					}
					else{
						((uint16_t *)ptr_result)[v * localw + u] = m_invalid_disp;
					}
				}
			}
		}
	}


	if (IsInterpolate){

		printf("Hist reject...........\n");

		uint16_t * ptr_result_u16 = (uint16_t *)ptr_result;

		float acc = 0;
		float UpThreshold = 0.010;
		float BottomThreshold = 0.005;
		float bottom = 0, up = maxDisparityG[lvl];
		for (int iid = 0; iid < maxDisparityG[lvl]; ++iid){
			acc += hist[iid] / fsum;
			if (acc > UpThreshold){
				bottom = iid;
				break;
			}
		}

		acc = 0;
		for (int iid = maxDisparityG[lvl] - 1; iid >= 0; --iid){
			acc += hist[iid] / fsum;
			if (acc > BottomThreshold){
				up = iid;
				break;
			}
		}

		for (int iid = 0; iid < localw*localh; ++iid){
			if (ptr_result_u16[iid] == m_invalid_disp){
				continue;
			}

			int idx = ptr_result_u16[iid] >> offset;

			if (idx < bottom){
				ptr_result_u16[iid] = m_invalid_disp;
			}

			if (idx > up){
				ptr_result_u16[iid] = m_invalid_disp;
			}
		}
	}

	return float(Valid) / float(Sum + 0.000001);
}

int StereoMatch::InitStereoMatch(int diparity_size, int input_depth, int output_depth, bool IsSubPixel)
{
    m_disparity_size = diparity_size;
    m_sgm_input_depth = input_depth;
    m_sgm_output_depth = output_depth;

	if (IsSubPixel){
		subPixelLevlG = 16;
	}
	else{
		subPixelLevlG = 1;
	}

    if (ptr_sgm) delete ptr_sgm;
    ptr_sgm = new sgm::StereoSGM(wG[0], hG[0], m_disparity_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA, 0, IsSubPixel);
    if (!ptr_sgm) printf("create sgm Object failed!\n");
    
    m_invalid_disp = output_depth == 8
                    ? static_cast< uint8_t>(ptr_sgm->get_invalid_disparity())
                    : static_cast<uint16_t>(ptr_sgm->get_invalid_disparity());

    if (ptr_d_left) delete ptr_d_left;
    if (ptr_d_right) delete ptr_d_right;
    if (ptr_d_disparity) delete ptr_d_disparity;

    m_input_bytes = int(input_depth * wG[0] * hG[0] / 8);
    m_output_bytes = int(output_depth * wG[0] * hG[0] / 8);
    ptr_d_left = new device_buffer(int(input_depth * wG[0] * hG[0] / 8));
    ptr_d_right = new device_buffer(int(input_depth * wG[0] * hG[0] / 8));
    ptr_d_disparity = new device_buffer(int(output_depth * wG[0] * hG[0] / 8));

    if (!(ptr_d_left && ptr_d_right && ptr_d_disparity)){
        printf("Create GPU buffer error!\n");
        return -1;
    }

	for (int lvl = 0; lvl < PyrLevelsUsedG; ++lvl){
		if (ptrLeftCensus[lvl]) free(ptrLeftCensus[lvl]);
		ptrLeftCensus[lvl] = (unsigned int *)malloc(sizeof(unsigned int)*wG[lvl]*hG[lvl]);
		if (ptrRightCensus[lvl]) free(ptrRightCensus[lvl]);
		ptrRightCensus[lvl] = (unsigned int *)malloc(sizeof(unsigned int)*wG[lvl]*hG[lvl]*4);

		if (ptrLeftCensus[lvl] == NULL || ptrRightCensus[lvl] == NULL){
			printf("CPU malloc error!\n");
			return -1;
		}	
		memset(ptrLeftCensus[lvl], 0, sizeof(unsigned int)*wG[lvl]*hG[lvl]);
		memset(ptrRightCensus[lvl], 0, sizeof(unsigned int)*wG[lvl]*hG[lvl]*4);
	}

	if (ptrRightCensusRaw) free(ptrRightCensusRaw);
	ptrRightCensusRaw = (unsigned int *)malloc(sizeof(unsigned int)*wG[0]*hG[0]);
	memset(ptrRightCensusRaw, 0, sizeof(unsigned int)*wG[0]*hG[0]);

	ptrCostVolumes = (unsigned int *)malloc(sizeof(unsigned int)*wG[0]*hG[0]*m_disparity_size);
	mptrFinalCost = (uint8_t *)malloc(8*wG[0]*hG[0]*m_disparity_size);

	mptrRightDisparity2 = (uint16_t * )malloc(hG[0]*wG[0]*sizeof(uint16_t));
	mptrAggregationCost = (uint16_t *)malloc(wG[0]*hG[0]*m_disparity_size*sizeof(uint16_t));

	memset(ptrCostVolumes, 0xff, 4*wG[0]*hG[0]*m_disparity_size);

	ptrCostVolumes2 = (unsigned int *)malloc(sizeof(unsigned int)*wG[0]*hG[0]*m_disparity_size);

	m_pattern_size = 9;
	ptr_d_cost = new device_buffer(4*wG[0]*hG[0]*m_disparity_size);

	ptrFinalCost = (uint8_t *)malloc(8*wG[0]*hG[0]*m_disparity_size);

    return m_invalid_disp;
}

void StereoMatch::ShowDisparity(void * ptr_disparity, std::string win_name, int lvl, bool flip){

	cv::Mat_<uint16_t> disparity(hG[lvl], wG[lvl]); 
	cv::Mat disparity_8u,disparity_color;

	memcpy(disparity.data, ptr_disparity, wG[lvl]*hG[lvl]*sizeof(uint16_t)); // uint16_t

	printf("Max:%d ShowMax:%d SubPixellvl:%d m_invalid_disp:%d\n",maxDisparityG[lvl], ShowMaxDisparity, subPixelLevlG, m_invalid_disp);

	disparity.convertTo(disparity_8u, CV_8U, 255. / maxDisparityG[lvl] / subPixelLevlG);
	cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
	disparity_color.setTo(cv::Scalar(0, 0, 0), disparity >= ShowMaxDisparity*subPixelLevlG);
	disparity_color.setTo(cv::Scalar(0, 0, 0), disparity == m_invalid_disp);
	disparity_color.setTo(cv::Scalar(0, 0, 0), disparity == 0);

	if (flip){
		cv::flip(disparity_color,disparity_color,0);
		cv::flip(disparity_color,disparity_color,1);
	}

	cv::imshow(win_name.c_str(), disparity_color);
 }


void StereoMatch::SetShowMaxDisparity(int max){
	if (max > 0){
		ShowMaxDisparity = max;
	}	
}


float getInterpolatedElement(const float* const mat, const float x, const float y, const int width)
{

	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	float dy = y - iy;
	float dxdy = dx*dy;
	const float* bp = mat +ix+iy*width;


	float res =   dxdy * bp[1+width]
				+ (dy-dxdy) * bp[width]
				+ (dx-dxdy) * bp[1]
				+ (1-dx-dy+dxdy) * bp[0];

	return res;
}


float getInterpolatedElementInt(const uint8_t* const mat, const float x, const float y, const int width)
{
	int ix = (int)x;
	int iy = (int)y;
	float dx = x - ix;
	float dy = y - iy;
	float dxdy = dx*dy;
	const uint8_t* bp = mat +ix+iy*width;


	float res =   dxdy * bp[1+width]
				+ (dy-dxdy) * bp[width]
				+ (dx-dxdy) * bp[1]
				+ (1-dx-dy+dxdy) * bp[0];

	return res;
}




int RSProjectionSubset2RTGrid2(Vec3 Angular, Vec3 Velocity, int start, int end, double * debuginfo, int lvl){

	int pyrScale = 1 << lvl;
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

	// Forward projection
    double OutBoader = 0, ProjFail = 0;
	double SumTryTimes = 0;
    double rightRow,leftRow,guessRightRow;
	float Ku,Kv;
    Mat33 Ridd;
    Vec3 tidd;
	Vec3 curLeftPoint, curRightPoint;
	double Threshold = 0.25;
    double u,v;
    int storeOffset = 0;
    float LastKu, LastKv;
	double newiDepth, idepth;
    for (int iid = start; iid < end; ++iid){
		leftRow = *(mptrAnchorLeftRawCoordi[lvl] + iid*2 + 1);
		int index = mptrAnchorV[lvl][iid/mAnchorWidth[lvl]] * wG[lvl] + mptrAnchorU[lvl][iid%mAnchorWidth[lvl]];
		storeOffset = index * maxDisparityG[lvl];

		mptrBaselineAcc[iid] = 0;
		mptrBaselineNum[iid] = 0;
		mptrEpipolarLength[iid] = 0;
		mptrBaselineAve[index] = 0;
		mptrdispscale[index] = 0;

		ptrTestInitialRightRow[iid] = 0;
		ptrTestFarLeftRow[iid] = 0;
		ptrTestCloseLeftRow[iid] = 0;

		if (mDisparityMask[lvl][index] == 0){
			continue;
		}

		// calc initial right row for the far space point
		guessRightRow = *(mptrInitialRightRawCoordi[lvl] + iid*2 + 1);
		curLeftPoint = Vec3(mptrAncherNormalzied[lvl][iid*3], mptrAncherNormalzied[lvl][iid*3 + 1], 1.0);

		// construct ray
		Mat33 lrinv = (Eigen::AngleAxisd(AngSpeed*leftRow, AngAxis).matrix()).inverse();
		Vec3 start = -leftRow*lrinv*Velocity;
		Vec3 inc = lrinv * curLeftPoint;

		// ptrTestCloseLeftRow[iid] = guessRightRow - leftRow;
		tidd = TLRG + (guessRightRow - leftRow - 128) * RLRG * Velocity;
		double CurBaseline = (RLRG.inverse()*tidd).norm();
		double ratio = CurBaseline / baselineG;

		mptrEpipolarLength[iid] = ratio;


		// double CloseKu = LastKu;
		// double CloseKv = LastKv;

		// if (CloseKu == -1 || DistKu == -1){
		// 	continue;
		// }

		// double UVDist = 1; // sqrtf((DistKv - LastKv)*(DistKv - LastKv) + (DistKu - LastKu)*(DistKu - LastKu)); // 1; // 
		// double AveBase = baseLine0 + baseLine1;

		// double TargetDisparity = 256;
		// int dispScale = 1;
		// if (ratio > 1.0){
		// 	TargetDisparity = 256;
		// 	dispScale = 1;
		// }
		// else if(ratio > 0.5){
		// 	TargetDisparity = 128;
		// 	dispScale = 2;
		// }
		// else if(ratio > 0.25){
		// 	TargetDisparity = 64;
		// 	dispScale = 4;
		// }
		// else{
		// 	TargetDisparity = 32;
		// 	dispScale = 8;
		// }

		// double TargetDisparity = 256;
		// int dispScale = 1;
		// if (ratio > 1.0){
		// 	TargetDisparity = 256;
		// 	dispScale = 1;
		// }
		// else if(ratio > 0.7){
		// 	TargetDisparity = 128;
		// 	dispScale = 2;
		// }
		// else {
		// 	TargetDisparity = 64;
		// 	dispScale = 4;
		// }

		double TargetDisparity = maxDisparityG[lvl];
		int dispScale = 1;

		mptrdispscale[index] = dispScale;
		mptrBaselineAve[index] = ratio;

		LastKu = LastKv = -1;
		// from far to close
		for (int kk = 0; kk < TargetDisparity; ++kk){

			idepth = invDepths[lvl][kk*dispScale];
			double depth = 1 / idepth;
			double lamba = (depth - start[2]) / inc[2];
			curLeftPoint = start + lamba * inc;

			// can be accelerated by precomputing
			int MaxTryTimes = 10;
			int tryTimes = 0;

			while(tryTimes < MaxTryTimes){
				Ridd = RLRG * Eigen::AngleAxisd(AngSpeed*guessRightRow, AngAxis).matrix();				
				tidd = TLRG + guessRightRow * RLRG * Velocity;

				curRightPoint = Ridd * curLeftPoint + tidd;

				// Just by a rotation!!
				u = curRightPoint[0] / curRightPoint[2];
				v = curRightPoint[1] / curRightPoint[2];            
				Ku = fxG[lvl] * u + cxG[lvl];
				Kv = fyG[lvl] * v + cyG[lvl];
				newiDepth = curRightPoint[2];

				if(!(Ku > mBorder[lvl] && Kv > mBorder[lvl] && Ku < (wG[lvl] - mBorder[lvl] - 1) && Kv < (hG[lvl] - mBorder[lvl] - 1) && newiDepth > 0)){
					OutBoader += 1;
					LastKu = -1;
					LastKv = -1;
					break;
				}

				// Interpolate 
				rightRow = getInterpolatedElement(ptrRightRemapYG, (Ku+0.5)*pyrScale-0.5, (Kv+0.5)*pyrScale-0.5, wG[0]);
				if (fabs(rightRow - guessRightRow) < Threshold){

					tidd = TLRG + guessRightRow * RLRG * Velocity - leftRow * Ridd * Velocity;
					mptrBaselineAcc[iid] += dispScale * (Ridd.inverse()*tidd).norm();
					mptrBaselineNum[iid] += dispScale;

					// if (LastKu > 0 && LastKv > 0){
					// 	mptrEpipolarLength[iid] += dispScale * sqrt((Ku - LastKu)*(Ku - LastKu) + (Kv - LastKv)*(Kv - LastKv));
					// }

					double delta;
					if (LastKu != -1){
						delta = acos(-1*(Ku-LastKu)/sqrt((Ku-LastKu)*(Ku-LastKu)+(Kv-LastKv)*(Kv-LastKv)));
					}

					////////////////////////////////////////////////////////////////////////
					for (int ww = 0; ww < dispScale; ++ww){
						ptrResCoordinate[(storeOffset + kk*dispScale + ww)*2] = Ku;
						ptrResCoordinate[(storeOffset + kk*dispScale + ww)*2 + 1] = Kv;

						if (LastKu != -1){
							if (Kv - LastKv > 0){
								ptrResOffset[(storeOffset+kk*dispScale+ww)*4] = int(round(12 * delta / 3.14159265)) % 12;
								ptrResOffset[(storeOffset+kk*dispScale+ww)*4 + 1] = int(round(8 * delta / 3.14159265)) % 8;
								ptrResOffset[(storeOffset+kk*dispScale+ww)*4 + 2] = int(round(6 * delta / 3.14159265)) % 6;
								ptrResOffset[(storeOffset+kk*dispScale+ww)*4 + 3] = int(round(4 * delta / 3.14159265)) % 4;
							}
							else{
								ptrResOffset[(storeOffset+kk*dispScale+ww)*4] = -(int(round(12 * delta / 3.14159265)) % 12);
								ptrResOffset[(storeOffset+kk*dispScale+ww)*4 + 1] = -(int(round(8 * delta / 3.14159265)) % 8);
								ptrResOffset[(storeOffset+kk*dispScale+ww)*4 + 2] = -(int(round(6 * delta / 3.14159265)) % 6);
								ptrResOffset[(storeOffset+kk*dispScale+ww)*4 + 3] = -(int(round(4 * delta / 3.14159265)) % 4);
							}
						}
					}

					LastKu = Ku;
					LastKv = Kv;
					break;
				}
				else{
					guessRightRow = rightRow;
				}

				tryTimes += 1;
				if (tryTimes >= MaxTryTimes){
					ProjFail += 1;
				}
			}
			SumTryTimes += (tryTimes + 1);
		}

		// Complement the gap postion for ptroffset
		for (int kk = maxDisparityG[lvl] - 2; kk >= 0; --kk){
			if (ptrResOffset[(storeOffset + kk)*4] == -64){
				ptrResOffset[(storeOffset + kk)*4] = ptrResOffset[(storeOffset + kk + 1)*4];
				ptrResOffset[(storeOffset + kk)*4 + 1] = ptrResOffset[(storeOffset + kk + 1)*4 + 1];
				ptrResOffset[(storeOffset + kk)*4 + 2] = ptrResOffset[(storeOffset + kk + 1)*4 + 2];
				ptrResOffset[(storeOffset + kk)*4 + 3] = ptrResOffset[(storeOffset + kk + 1)*4 + 3];
			}
		}

		if (mptrBaselineNum[iid] > 0){
			mptrBaselineAcc[iid] = mptrBaselineAcc[iid] / mptrBaselineNum[iid];
		}
    }

	*(debuginfo + 0) = SumTryTimes;
	*(debuginfo + 1) = OutBoader;
	*(debuginfo + 2) = ProjFail;

	return 0;
}



int InterpolateProj2(int start, int end, int lvl){

	for (int iid = start; iid < end; ++iid){
		int xstart = mptrAnchorU[lvl][iid % (mAnchorWidth[lvl] - 1)];
		int ystart = mptrAnchorV[lvl][iid / (mAnchorWidth[lvl] - 1)];
		int xend = mptrAnchorU[lvl][iid % (mAnchorWidth[lvl] - 1) + 1];
		int yend = mptrAnchorV[lvl][iid / (mAnchorWidth[lvl] - 1) + 1];

		// printf("%d %d %d %d\n", xstart, ystart, xend, yend);

		uint8_t scale = mptrdispscale[ystart*wG[lvl] + xstart];
		scale = scale < mptrdispscale[ystart*wG[lvl] + xend] ? scale : mptrdispscale[ystart*wG[lvl] + xend];
		scale = scale < mptrdispscale[yend*wG[lvl] + xstart] ? scale : mptrdispscale[yend*wG[lvl] + xstart];
		scale = scale < mptrdispscale[yend*wG[lvl] + xend] ? scale : mptrdispscale[yend*wG[lvl] + xend];

		double r0 = mptrBaselineAve[ystart*wG[lvl] + xstart];
		double r1 = mptrBaselineAve[ystart*wG[lvl] + xend];
		double r2 = mptrBaselineAve[yend*wG[lvl] + xstart];
		double r3 = mptrBaselineAve[yend*wG[lvl] + xend];

		for (int yy = ystart; yy < yend; ++yy){
			for (int xx = xstart; xx < xend; ++xx){
				if (xx == xstart && yy == ystart) continue;
				mptrdispscale[yy*wG[lvl] + xx] = scale;
				if (scale > 0){
					mptrBaselineAve[yy*wG[lvl] + xx] = ptrInterpolateCoeffsTL[lvl][yy*wG[lvl] + xx] * r0 + 
												     ptrInterpolateCoeffsTR[lvl][yy*wG[lvl] + xx] * r1 + 
												     ptrInterpolateCoeffsBL[lvl][yy*wG[lvl] + xx] * r2 + 
												     ptrInterpolateCoeffsBR[lvl][yy*wG[lvl] + xx] * r3;

				}
				else{
					mptrBaselineAve[yy*wG[lvl] + xx] = 0;
				}
			}
		}

		if (!mDisparityMask[lvl][ystart*wG[lvl] + xstart] || 
		    !mDisparityMask[lvl][ystart*wG[lvl] + xend] ||
			!mDisparityMask[lvl][yend*wG[lvl] + xstart] ||
			!mDisparityMask[lvl][yend*wG[lvl] + xend]){

			continue;
		}

		for (int dis = 0; dis < maxDisparityG[lvl]; ++dis){
			float u0 = ptrResCoordinate[((ystart*wG[lvl] + xstart)*maxDisparityG[lvl] + dis)*2];
			float u1 = ptrResCoordinate[((ystart*wG[lvl] + xend)*maxDisparityG[lvl] + dis)*2];
			float u2 = ptrResCoordinate[((yend*wG[lvl] + xstart)*maxDisparityG[lvl] + dis)*2];
			float u3 = ptrResCoordinate[((yend*wG[lvl] + xend)*maxDisparityG[lvl] + dis)*2];

			if (u0 < 0 || u1 < 0 || u2 < 0 || u3 < 0){
				for (int yy = ystart; yy < yend; ++yy){
					for (int xx = xstart; xx < xend; ++xx){
						if (xx == xstart && yy == ystart) continue;
						int storeOffset = (yy*wG[lvl] + xx)*maxDisparityG[lvl] + dis;
						ptrResCoordinate[storeOffset*2]  = -1;
						ptrResCoordinate[storeOffset*2 + 1]  = -1;

						ptrResOffset[storeOffset*4] = -64;
						ptrResOffset[storeOffset*4 + 1] = -64;
						ptrResOffset[storeOffset*4 + 2] = -64;
						ptrResOffset[storeOffset*4 + 3] = -64;
					}
				}
				continue;
			}

			float o00 = ptrResOffset[((ystart*wG[lvl] + xstart)*maxDisparityG[lvl] + dis)*4];
			float o01 = ptrResOffset[((ystart*wG[lvl] + xend)*maxDisparityG[lvl] + dis)*4];
			float o02 = ptrResOffset[((yend*wG[lvl] + xstart)*maxDisparityG[lvl] + dis)*4];
			float o03 = ptrResOffset[((yend*wG[lvl] + xend)*maxDisparityG[lvl] + dis)*4];

			if (o00 == -64 || o01 == -64 || o02 == -64 || o03 == -64){
				for (int yy = ystart; yy < yend; ++yy){
					for (int xx = xstart; xx < xend; ++xx){
						if (xx == xstart && yy == ystart) continue;
						int storeOffset = (yy*wG[lvl] + xx)*maxDisparityG[lvl] + dis;
						ptrResCoordinate[storeOffset*2]  = -1;
						ptrResCoordinate[storeOffset*2 + 1]  = -1;

						ptrResOffset[storeOffset*4] = -64;
						ptrResOffset[storeOffset*4 + 1] = -64;
						ptrResOffset[storeOffset*4 + 2] = -64;
						ptrResOffset[storeOffset*4 + 3] = -64;
					}
				}
				continue;
			}

			float o10 = ptrResOffset[((ystart*wG[lvl] + xstart)*maxDisparityG[lvl] + dis)*4 + 1];
			float o11 = ptrResOffset[((ystart*wG[lvl] + xend)*maxDisparityG[lvl] + dis)*4 + 1];
			float o12 = ptrResOffset[((yend*wG[lvl] + xstart)*maxDisparityG[lvl] + dis)*4 + 1];
			float o13 = ptrResOffset[((yend*wG[lvl] + xend)*maxDisparityG[lvl] + dis)*4 + 1];

			float o20 = ptrResOffset[((ystart*wG[lvl] + xstart)*maxDisparityG[lvl] + dis)*4 + 2];
			float o21 = ptrResOffset[((ystart*wG[lvl] + xend)*maxDisparityG[lvl] + dis)*4 + 2];
			float o22 = ptrResOffset[((yend*wG[lvl] + xstart)*maxDisparityG[lvl] + dis)*4 + 2];
			float o23 = ptrResOffset[((yend*wG[lvl] + xend)*maxDisparityG[lvl] + dis)*4 + 2];

			float o30 = ptrResOffset[((ystart*wG[lvl] + xstart)*maxDisparityG[lvl] + dis)*4 + 3];
			float o31 = ptrResOffset[((ystart*wG[lvl] + xend)*maxDisparityG[lvl] + dis)*4 + 3];
			float o32 = ptrResOffset[((yend*wG[lvl] + xstart)*maxDisparityG[lvl] + dis)*4 + 3];
			float o33 = ptrResOffset[((yend*wG[lvl] + xend)*maxDisparityG[lvl] + dis)*4 + 3];

			float v0 = ptrResCoordinate[((ystart*wG[lvl] + xstart)*maxDisparityG[lvl] + dis)*2 + 1];
			float v1 = ptrResCoordinate[((ystart*wG[lvl] + xend)*maxDisparityG[lvl] + dis)*2 + 1];
			float v2 = ptrResCoordinate[((yend*wG[lvl] + xstart)*maxDisparityG[lvl] + dis)*2 + 1];
			float v3 = ptrResCoordinate[((yend*wG[lvl] + xend)*maxDisparityG[lvl] + dis)*2 + 1];

			for (int yy = ystart; yy < yend; ++yy){
				for (int xx = xstart; xx < xend; ++xx){
					int storeOffset = (yy*wG[lvl] + xx)*maxDisparityG[lvl] + dis;
					ptrResCoordinate[2*storeOffset] = ptrInterpolateCoeffsTL[lvl][yy*wG[lvl] + xx] * u0 + 
												  ptrInterpolateCoeffsTR[lvl][yy*wG[lvl] + xx] * u1 + 
												  ptrInterpolateCoeffsBL[lvl][yy*wG[lvl] + xx] * u2 + 
												  ptrInterpolateCoeffsBR[lvl][yy*wG[lvl] + xx] * u3;

					ptrResCoordinate[2*storeOffset + 1] = ptrInterpolateCoeffsTL[lvl][yy*wG[lvl] + xx] * v0 + 
													  ptrInterpolateCoeffsTR[lvl][yy*wG[lvl] + xx] * v1 + 
													  ptrInterpolateCoeffsBL[lvl][yy*wG[lvl] + xx] * v2 + 
													  ptrInterpolateCoeffsBR[lvl][yy*wG[lvl] + xx] * v3;

					ptrResOffset[4*storeOffset] = (char)(ptrInterpolateCoeffsTL[lvl][yy*wG[lvl] + xx] * o00 + 
												  ptrInterpolateCoeffsTR[lvl][yy*wG[lvl] + xx] * o01 + 
												  ptrInterpolateCoeffsBL[lvl][yy*wG[lvl] + xx] * o02 + 
												  ptrInterpolateCoeffsBR[lvl][yy*wG[lvl] + xx] * o03);

					ptrResOffset[4*storeOffset + 1] = (char)(ptrInterpolateCoeffsTL[lvl][yy*wG[lvl] + xx] * o10 + 
													  ptrInterpolateCoeffsTR[lvl][yy*wG[lvl] + xx] * o11 + 
													  ptrInterpolateCoeffsBL[lvl][yy*wG[lvl] + xx] * o12 + 
													  ptrInterpolateCoeffsBR[lvl][yy*wG[lvl] + xx] * o13);

					ptrResOffset[4*storeOffset + 2] = (char)(ptrInterpolateCoeffsTL[lvl][yy*wG[lvl] + xx] * o20 + 
												  	  ptrInterpolateCoeffsTR[lvl][yy*wG[lvl] + xx] * o21 + 
												  	  ptrInterpolateCoeffsBL[lvl][yy*wG[lvl] + xx] * o22 + 
												  	  ptrInterpolateCoeffsBR[lvl][yy*wG[lvl] + xx] * o23);

					ptrResOffset[4*storeOffset + 3] = (char)(ptrInterpolateCoeffsTL[lvl][yy*wG[lvl] + xx] * o30 + 
													  ptrInterpolateCoeffsTR[lvl][yy*wG[lvl] + xx] * o31 + 
													  ptrInterpolateCoeffsBL[lvl][yy*wG[lvl] + xx] * o32 + 
													  ptrInterpolateCoeffsBR[lvl][yy*wG[lvl] + xx] * o33);
				}
			}

			int baseindex = dis*wG[lvl]*hG[lvl]*2;
			int top_v = 0, top_u, bottom_v = 0, bottom_u;
			float deltaU = -1, deltaY, CurYStart;
			float vv0, vv2, uu0, uu2;
			vv0 = v0;
			vv2 = v2;
			uu0 = u0;
			uu2 = u2;
			if (vv0 < vv2){
				top_v = int(vv0 + 0.5);
				top_u = int(uu0 + 0.5);
				bottom_v = int(vv2 + 0.5);
				bottom_u = int(uu2 + 0.5);
				deltaU = float(bottom_u - top_u) / float(bottom_v - top_v);
				deltaY =  float(yend - ystart) / float(bottom_v - top_v);
				CurYStart = ystart;
			}
			else if(vv0 > vv2){
				bottom_v = int(vv0 + 0.5);
				bottom_u = int(uu0 + 0.5);
				top_v = int(vv2 + 0.5);
				top_u = int(uu2 + 0.5);
				deltaU = float(bottom_u - top_u) / float(bottom_v - top_v);
				deltaY =  float(ystart - yend) / float(bottom_v - top_v);
				CurYStart = yend;
			}

			for (int kk = top_v; kk < bottom_v; ++kk){
				int curU = int(top_u + deltaU*(kk - top_v) + 0.5);
				if (curU < 0 || curU > wG[0] - 1){
					continue;
				}
				ptrResCoordinateR2L[baseindex + (kk*wG[lvl] + curU)*2] = xstart; // LeftU
				ptrResCoordinateR2L[baseindex + (kk*wG[lvl] + curU)*2 + 1] = CurYStart + deltaY*(kk - top_v); // LeftV
			}

			top_v = 0, bottom_v = 0;
			deltaU = -1;

			vv0 = v1;
			vv2 = v3;
			uu0 = u1;
			uu2 = u3;
			if (vv0 < vv2){
				top_v = int(vv0 + 0.5);
				top_u = int(uu0 + 0.5);
				bottom_v = int(vv2 + 0.5);
				bottom_u = int(uu2 + 0.5);
				deltaU = float(bottom_u - top_u) / float(bottom_v - top_v);
				deltaY =  float(yend - ystart) / float(bottom_v - top_v);
				CurYStart = ystart;
			}
			else if(vv0 > vv2){
				bottom_v = int(vv0 + 0.5);
				bottom_u = int(uu0 + 0.5);
				top_v = int(vv2 + 0.5);
				top_u = int(uu2 + 0.5);
				deltaU = float(bottom_u - top_u) / float(bottom_v - top_v);
				deltaY =  float(ystart - yend) / float(bottom_v - top_v);
				CurYStart = yend;
			}

			for (int kk = top_v; kk < bottom_v; ++kk){
				int curU = int(top_u + deltaU*(kk - top_v) + 0.5);
				if (curU < 0 || curU > wG[lvl] - 1){
					continue;
				}
				ptrResCoordinateR2L[baseindex + (kk*wG[lvl] + curU)*2] = xstart; // LeftU
				ptrResCoordinateR2L[baseindex + (kk*wG[lvl] + curU)*2 + 1] = CurYStart + deltaY*(kk - top_v); // LeftV
			}

		}
	}

	return 0;
}

int InterpolateProjR2L(int start, int end, int lvl){
	for (int iid = start; iid < end; ++iid){
		int baseindex = iid*wG[lvl]*hG[lvl]*2;
		for (int vv = 0; vv < hG[lvl]; ++vv){
			int begin = 0;
			int rowindex = vv * wG[lvl] * 2;
			while (begin < wG[lvl] && begin >= 0){
				int start = -1, end = -1;
				for (int uu = begin; uu < wG[lvl]; ++uu){
					if (ptrResCoordinateR2L[baseindex + rowindex + uu*2] > 0){
						start = uu;
						break;
					}
				}
				if (start > 0){
					for (int uu = start + 1; uu < wG[lvl]; ++uu){
						if (ptrResCoordinateR2L[baseindex + rowindex + uu*2] > 0){
							end = uu;
							break;
						}
					}
				}

				if (start > 0 && end > 0 && end - start < 3* mGridSize[lvl]){
					// Interpolate
					float lu = ptrResCoordinateR2L[baseindex + rowindex + start*2];
					float lv = ptrResCoordinateR2L[baseindex + rowindex + start*2 + 1];
					float ru = ptrResCoordinateR2L[baseindex + rowindex + end*2];
					float rv = ptrResCoordinateR2L[baseindex + rowindex + end*2 + 1];
					float length = end - start;
					for (int kk = start + 1; kk < end; ++kk){
						ptrResCoordinateR2L[baseindex + rowindex + kk*2] = ru*(kk-start)/length + lu*(end-kk)/length;
						ptrResCoordinateR2L[baseindex + rowindex + kk*2 + 1] = rv*(kk-start)/length + lv*(end-kk)/length;
					}
				}

				begin = end;
			}
		}
				

	}

	return 0;
}



// Show RS Projection result
int StereoMatch::CheckCorrespondence(int lvl, bool left2right){
	cv::Mat LeftUnDistTarget(hG[lvl], wG[lvl], CV_8U, mleftFrame->grayPyr[lvl]);
	cv::Mat RightUnDistTarget(hG[lvl], wG[lvl], CV_8U, mrightFrame->grayPyr[lvl]);
    cv::cvtColor(LeftUnDistTarget, LeftUnDistTarget, cv::COLOR_GRAY2BGR);
    cv::cvtColor(RightUnDistTarget, RightUnDistTarget, cv::COLOR_GRAY2BGR);
	int scale = 1 << lvl;
	int step = 100 / scale;

	if (left2right){
		for(int iid = wG[lvl]*3; iid < hG[lvl]*wG[lvl]; iid+=step){
			if (ptrResCoordinate[iid*maxDisparityG[lvl]*2] > 0){
				cv::Mat left = LeftUnDistTarget.clone();
				cv::Mat right = RightUnDistTarget.clone();
				cv::Point2f point;
				point.x = iid % wG[lvl];
				point.y = iid / wG[lvl];
				cout << "leftPoint:" << point << std::endl;
				cv::circle(left, point, 3, cv::Scalar(0,0,255),2);
				for (int kk = 0; kk < maxDisparityG[lvl]; ++kk){
					point.x = ptrResCoordinate[((iid*maxDisparityG[lvl]) + kk)*2];
					point.y = ptrResCoordinate[((iid*maxDisparityG[lvl]) + kk)*2 + 1];
					cout << "rightPoint:" << point << std::endl;
					cv::circle(right, point, 0.5, cv::Scalar(0,0,255),0.5);
				}
				cv::imshow("left", left);
				cv::imshow("right", right);
				cv::waitKey(0);
			}
		}
	}
	else{
		for(int iid = wG[lvl]*3; iid < hG[lvl]*wG[lvl]; iid+=step){
			if (ptrResCoordinateR2L[iid*maxDisparityG[lvl]*2] > 0){
				cv::Mat left = LeftUnDistTarget.clone();
				cv::Mat right = RightUnDistTarget.clone();
				cv::Point2f point;
				point.x = iid % wG[lvl];
				point.y = iid / wG[lvl];
				cout << "rightPoint:" << point << std::endl;
				cv::circle(right, point, 3, cv::Scalar(0,0,255),2);
				for (int kk = 0; kk < maxDisparityG[lvl]; ++kk){
					point.x = ptrResCoordinateR2L[(kk*hG[lvl]*wG[lvl] + iid)*2];
					point.y = ptrResCoordinateR2L[(kk*hG[lvl]*wG[lvl] + iid)*2 + 1];
					cout << "leftPoint:" << point << std::endl;
					cv::circle(left, point, 0.5, cv::Scalar(0,0,255),0.5);
				}
				cv::imshow("left", left);
				cv::imshow("right", right);
				cv::waitKey(0);
			}
		}
	}

	return 0;
}


int StereoMatch::CheckLRDisparity(uint16_t * leftRaw, uint16_t * rightRaw, int lvl){
	cv::Mat LeftDisparity(hG[lvl], wG[lvl], CV_16U, leftRaw);
	cv::Mat RightDisparity(hG[lvl], wG[lvl], CV_16U, rightRaw);

	cv::Mat left_u8, right_u8, left_color, right_color;
	LeftDisparity.convertTo(left_u8, CV_8U, 255.0 / maxDisparityG[lvl] / subPixelLevlG);
	RightDisparity.convertTo(right_u8, CV_8U, 255.0 / maxDisparityG[lvl] / subPixelLevlG);

	cv::applyColorMap(left_u8, left_color, cv::COLORMAP_JET);
	left_color.setTo(cv::Scalar(0, 0, 0), LeftDisparity >= maxDisparityG[lvl]*subPixelLevlG);
	left_color.setTo(cv::Scalar(0, 0, 0), LeftDisparity == invalidDispG);
	left_color.setTo(cv::Scalar(0, 0, 0), LeftDisparity == 0);

	cv::applyColorMap(right_u8, right_color, cv::COLORMAP_JET);
	right_color.setTo(cv::Scalar(0, 0, 0), RightDisparity >= maxDisparityG[lvl]*subPixelLevlG);
	right_color.setTo(cv::Scalar(0, 0, 0), RightDisparity == invalidDispG);
	right_color.setTo(cv::Scalar(0, 0, 0), RightDisparity == 0);

	// int scale = 1 << lvl;

	for(int iid = wG[lvl]*3; iid < hG[lvl]*wG[lvl]; iid+=1){
		if (ptrResCoordinate[iid*maxDisparityG[lvl]*2] > 0){
			if (mptrdispscale[iid] == 0){
				continue;
			}
			uint16_t leftDisp = leftRaw[iid];
			if (leftDisp == invalidDispG){
				continue;
			}

			cv::Mat left = left_color.clone();
			cv::Mat right = right_color.clone();
			cv::Point2f leftpoint, rightpoint;
			leftpoint.x = iid % wG[lvl];
			leftpoint.y = iid / wG[lvl];

			uint16_t kk = leftDisp >> 4;

			rightpoint.x = ptrResCoordinate[((iid*maxDisparityG[lvl]) + kk)*2];
			rightpoint.y = ptrResCoordinate[((iid*maxDisparityG[lvl]) + kk)*2 + 1];
			if (rightpoint.x < 0){
				continue;
			}
			cv::circle(left, leftpoint, 3, cv::Scalar(0,0,255),2);
			cv::circle(right, rightpoint, 3, cv::Scalar(0,0,255),2);

			uint16_t rightDisp = rightRaw[int(rightpoint.y + 0.5)*wG[lvl] + int(rightpoint.x + 0.5)];
			if (rightDisp == invalidDispG){
				continue;
			}
			rightDisp = rightDisp >> 4;

			if (fabs(float(kk) - float(rightDisp)) > 1){
				cout << "leftPoint:" << leftpoint << std::endl;
				printf("left disparity:%d\n", kk);
				cout << "rightPoint:" << rightpoint << std::endl;
				printf("right disparity:%d\n", rightDisp);
				cv::imshow("left", left);
				cv::imshow("right", right);
				cv::waitKey(0);
				leftRaw[iid] = invalidDispG;
				left_color.at<uint8_t>(iid*3 + 0) = 0;
				left_color.at<uint8_t>(iid*3 + 1) = 0;
				left_color.at<uint8_t>(iid*3 + 2) = 0;
			}
		}
	}

	return 0;
}


// Compute coeffiecients for later interpolate
int CalcGridInterpolateCoeffs(int Border, int GridSize, int lvl){

	if (ptrInterpolateCoeffsTL[lvl]){
		free(ptrInterpolateCoeffsTL[lvl]);
	}
	ptrInterpolateCoeffsTL[lvl] = (float *)malloc(sizeof(float)*wG[lvl]*hG[lvl]);

	if (ptrInterpolateCoeffsTR[lvl]){
		free(ptrInterpolateCoeffsTR[lvl]);
	}
	ptrInterpolateCoeffsTR[lvl] = (float *)malloc(sizeof(float)*wG[lvl]*hG[lvl]);

	if (ptrInterpolateCoeffsBL[lvl]){
		free(ptrInterpolateCoeffsBL[lvl]);
	}
	ptrInterpolateCoeffsBL[lvl] = (float *)malloc(sizeof(float)*wG[lvl]*hG[lvl]);

	if (ptrInterpolateCoeffsBR[lvl]){
		free(ptrInterpolateCoeffsBR[lvl]);
	}
	ptrInterpolateCoeffsBR[lvl] = (float *)malloc(sizeof(float)*wG[lvl]*hG[lvl]);

	// Interpolate coeffiecients can be precomputed
	for (int vv = Border; vv < hG[lvl]; vv+=GridSize){
		for (int uu = Border; uu < wG[lvl]; uu+=GridSize){
			double xstart = uu, ystart = vv, xend, yend;
			xend = (uu + GridSize < wG[lvl]) ? (uu + GridSize) : (wG[lvl] - Border - 1);
			yend = (vv + GridSize < hG[lvl]) ? (vv + GridSize) : (hG[lvl] - Border - 1);

			for (int yy = ystart; yy < yend; ++yy){
				for (int xx = xstart; xx < xend; ++xx){
					double r1 = (xx - xstart) / (xend - xstart);
					double r2 = (yy - ystart) / (yend - ystart);
					// assign value to 4 channels
					ptrInterpolateCoeffsTL[lvl][yy*wG[lvl] + xx] = (1 - r1)*(1 - r2);
					ptrInterpolateCoeffsTR[lvl][yy*wG[lvl] + xx] = r1*(1 - r2);
					ptrInterpolateCoeffsBL[lvl][yy*wG[lvl] + xx] = (1 - r1)*r2;
					ptrInterpolateCoeffsBR[lvl][yy*wG[lvl] + xx] = r1*r2;
				}
			}
		}
	}

	return 0;
}

int StereoMatch::InitRSProjection(int GridSize, int Border){
	// Set
	for (int lvl = 0; lvl < PyrLevelsUsedG; ++lvl){
		mGridSize[lvl] = GridSize - lvl*2;
		if (mGridSize[lvl] <= 0){
			mGridSize[lvl] = 1;
		}
		mBorder[lvl] = Border;
		if (mBorder[lvl] > 0 || mGridSize[lvl] > 0){
			if ((hG[lvl] - 2*mBorder[lvl]) % mGridSize[lvl] == 0){
				mAnchorHeight[lvl] = (hG[lvl] - 2*mBorder[lvl]) / mGridSize[lvl] + 1;	
			}
			else{
				mAnchorHeight[lvl] = ((hG[lvl] - 2*mBorder[lvl]) / mGridSize[lvl]) + 2;
			}

			if ((wG[lvl] - 2*mBorder[lvl]) % mGridSize[lvl] == 0){
				mAnchorWidth[lvl] = (wG[lvl] - 2*mBorder[lvl]) / mGridSize[lvl] + 1;
			}
			else{
				mAnchorWidth[lvl] = (wG[lvl] - 2*mBorder[lvl]) / mGridSize[lvl] + 2;
			}
		}
		else{
			mAnchorHeight[lvl] = hG[lvl];
			mAnchorWidth[lvl] = wG[lvl];
		}

		if (mptrAncherNormalzied[lvl]){
			free(mptrAncherNormalzied[lvl]);
		}
		mptrAncherNormalzied[lvl] = (double *)malloc(mAnchorHeight[lvl]*mAnchorWidth[lvl]*sizeof(double)*3);
		if (mptrAnchorLeftRawCoordi[lvl]){
			free(mptrAnchorLeftRawCoordi[lvl]);
		}
		mptrAnchorLeftRawCoordi[lvl] = (double *)malloc(mAnchorHeight[lvl]*mAnchorWidth[lvl]*sizeof(double)*2);
		if (mptrInitialRightRawCoordi[lvl]){
			free(mptrInitialRightRawCoordi[lvl]);
		}
		mptrInitialRightRawCoordi[lvl] = (double *)malloc(mAnchorHeight[lvl]*mAnchorWidth[lvl]*sizeof(double)*2);

		if (mptrInitialRightTargetCoordi[lvl]){
			free(mptrInitialRightTargetCoordi[lvl]);
		}
		mptrInitialRightTargetCoordi[lvl] = (double *)malloc(mAnchorHeight[lvl]*mAnchorWidth[lvl]*sizeof(double)*2);

		if (mptrAnchorU[lvl]){
			free(mptrAnchorU[lvl]);
			free(mptrAnchorV[lvl]);
		}
		mptrAnchorU[lvl] = (int *)malloc(sizeof(int)*mAnchorWidth[lvl]);
		mptrAnchorV[lvl] = (int *)malloc(sizeof(int)*mAnchorHeight[lvl]);

		for (int uu = 0; uu < mAnchorWidth[lvl] - 1; ++uu){
			mptrAnchorU[lvl][uu] = mBorder[lvl] + mGridSize[lvl] * uu;
		}
		mptrAnchorU[lvl][mAnchorWidth[lvl] - 1] = wG[lvl] - mBorder[lvl] - 1;
		for (int vv = 0; vv < mAnchorHeight[lvl] - 1; ++vv){
			mptrAnchorV[lvl][vv] = mBorder[lvl] + mGridSize[lvl] * vv;
		}
		mptrAnchorV[lvl][mAnchorHeight[lvl] - 1] = hG[lvl] - mBorder[lvl] - 1;


		if (mGridSize[lvl] != 1){
			CalcGridInterpolateCoeffs(mBorder[lvl], mGridSize[lvl], lvl);
		}

		for (int vv = 0; vv < mAnchorHeight[lvl]; ++vv){
			for (int uu = 0; uu < mAnchorWidth[lvl]; ++uu){
				int index = (vv*mAnchorWidth[lvl] + uu)*3;
				mptrAncherNormalzied[lvl][index + 0] = mptrAnchorU[lvl][uu];
				mptrAncherNormalzied[lvl][index + 1] = mptrAnchorV[lvl][vv];
				mptrAncherNormalzied[lvl][index + 2] = 1.0;
			}
		}

		// Notice
		for (int ii = 1; ii <= maxDisparityG[lvl]; ++ii){
			invDepths[lvl][ii-1] = ii / (fxG[lvl] * baselineG * baselineScaleG);
		}

		double ptrLeftR[9] = {0};
		ptrLeftR[0] = 1.0;
		ptrLeftR[4] = 1.0;
		ptrLeftR[8] = 1.0;
		double ptrLeftT[3] = {0};

		cv::Mat CvLeftR(3,3,CV_64F,ptrLeftR);
		cv::Mat CvLeftT(3,1,CV_64F,ptrLeftT);

		cv::Mat AnchorNormalzied(mAnchorWidth[lvl]*mAnchorHeight[lvl], 3, CV_64F, mptrAncherNormalzied[lvl]);
		AnchorNormalzied = AnchorNormalzied.t();
		cv::Mat AnchorLeftRawCoordi(mAnchorWidth[lvl]*mAnchorHeight[lvl], 2, CV_64F);
		cv::Mat InitialRightRawCoordi(mAnchorWidth[lvl]*mAnchorHeight[lvl], 2, CV_64F);
		cv::Mat InitialRightTargetCoordi(mAnchorWidth[lvl]*mAnchorHeight[lvl], 2, CV_64F);
		AnchorNormalzied = cvKG[lvl].inv() * AnchorNormalzied;

		AnchorNormalzied = AnchorNormalzied.t();
		memcpy(mptrAncherNormalzied[lvl], AnchorNormalzied.ptr<uchar>(0), 3*sizeof(double)*mAnchorWidth[lvl]*mAnchorHeight[lvl]);

		cv::projectPoints(AnchorNormalzied, CvLeftR, CvLeftT, cvK0G, cvDistCoef0, AnchorLeftRawCoordi);
		memcpy(mptrAnchorLeftRawCoordi[lvl], AnchorLeftRawCoordi.ptr<uchar>(0), mAnchorHeight[lvl]*mAnchorWidth[lvl]*sizeof(double)*2);

		// cout << cvRLRG << std::endl;
		cv::projectPoints(AnchorNormalzied, cvRLRG, cv::Mat(3,1,CV_64F,0.0), cvK1G, cvDistCoef1, InitialRightRawCoordi);
		memcpy(mptrInitialRightRawCoordi[lvl], InitialRightRawCoordi.ptr<uchar>(0), mAnchorHeight[lvl]*mAnchorWidth[lvl]*sizeof(double)*2);

		cv::Mat DistCoef(1,5,CV_64F,0.0);
		cv::projectPoints(AnchorNormalzied, cvRLRG, cv::Mat(3,1,CV_64F,0.0), cvKG[lvl], DistCoef, InitialRightTargetCoordi);
		memcpy(mptrInitialRightTargetCoordi[lvl], InitialRightTargetCoordi.ptr<uchar>(0), mAnchorHeight[lvl]*mAnchorWidth[lvl]*sizeof(double)*2);

		cv::Mat coordi(wG[lvl]*hG[lvl], 3, CV_64F);
		for (int ii = 0; ii < wG[lvl]*hG[lvl]; ++ii){
			double * ptr = (double *)coordi.ptr(ii);
			ptr[0] = ii % wG[lvl];
			ptr[1] = ii / wG[lvl];
			ptr[2] = 1.0;
		}
		
		coordi = coordi.t();
		cv::Mat rightCoordi = cvRLRG * cvKiG[lvl] * coordi;
		cv::Mat leftCoordi = cvRLRG.inv() * cvKiG[lvl] * coordi;
		rightCoordi = rightCoordi.t();
		leftCoordi = leftCoordi.t();
		for (int ii = 0; ii < wG[lvl]*hG[lvl]; ++ii){
			double * ptr = (double *)rightCoordi.ptr(ii);
			ptr[0] = ptr[0] / ptr[2];
			ptr[1] = ptr[1] / ptr[2];
			ptr[2] = 1.0;

			ptr = (double *)leftCoordi.ptr(ii);
			ptr[0] = ptr[0] / ptr[2];
			ptr[1] = ptr[1] / ptr[2];
			ptr[2] = 1.0;
		}
		rightCoordi = rightCoordi.t();
		rightCoordi = cvKG[lvl] * rightCoordi;
		rightCoordi = rightCoordi.t();

		leftCoordi = leftCoordi.t();
		leftCoordi = cvKG[lvl] * leftCoordi;
		leftCoordi = leftCoordi.t();

		// Judge safe projection
		for (int row = 0; row < hG[lvl]; ++row){
			for (int col = 0; col < wG[lvl]; ++col){

				double RightTargetCol = rightCoordi.at<double>(row*wG[lvl] + col, 0);
				double RightTargetRow = rightCoordi.at<double>(row*wG[lvl] + col, 1);

				if (RightTargetRow < mBorder[lvl] || RightTargetRow > (hG[lvl] - mBorder[lvl] - 1) ||
					RightTargetCol < mBorder[lvl] || RightTargetCol > (wG[lvl] - mBorder[lvl] - 1)){
					mDisparityMask[lvl][row*wG[lvl] + col] = 0;
				}


				double LeftTargetCol = leftCoordi.at<double>(row*wG[lvl] + col, 0);
				double LeftTargetRow = leftCoordi.at<double>(row*wG[lvl] + col, 1);

				if (LeftTargetRow < mBorder[lvl] || LeftTargetRow > (hG[lvl] - mBorder[lvl] - 1) ||
					LeftTargetCol < mBorder[lvl] || LeftTargetCol > (wG[lvl] - mBorder[lvl] - 1)){
					Right2LeftMask[lvl][row*wG[lvl] + col] = 0;
				}

			}
		}
	}


	if (ptrResCoordinate){
		free(ptrResCoordinate);
	}
	ptrResCoordinate = (float *)malloc(2*sizeof(float)*wG[0]*hG[0]*maxDisparityG[0]);
	if (ptrResCoordinateR2L){
		free(ptrResCoordinateR2L);
	}
	ptrResCoordinateR2L = (float *)malloc(2*sizeof(float)*wG[0]*hG[0]*maxDisparityG[0]);
	if (ptrResOffset){
		free(ptrResOffset);
	}
	ptrResOffset = (char *)malloc(4*sizeof(char)*wG[0]*hG[0]*maxDisparityG[0]);


	////////////////////////
	mptrBaselineAcc = (double *)malloc(sizeof(double)*mAnchorWidth[0]*mAnchorHeight[0]);
	mptrBaselineNum = (double *)malloc(sizeof(double)*mAnchorWidth[0]*mAnchorHeight[0]);
	memset(mptrBaselineAcc, 0, sizeof(double)*mAnchorWidth[0]*mAnchorHeight[0]);
	memset(mptrBaselineNum, 0, sizeof(double)*mAnchorWidth[0]*mAnchorHeight[0]);

	mptrBaselineAve = (double *)malloc(sizeof(double)*wG[0]*hG[0]);
	mptrdispscale = (uint8_t *)malloc(sizeof(uint8_t)*wG[0]*hG[0]);
	memset(mptrBaselineAve, 0, sizeof(double)*wG[0]*hG[0]);
	memset(mptrdispscale, 0, sizeof(uint8_t)*wG[0]*hG[0]);

	ptrTestInitialRightRow = (float *)malloc(sizeof(float)*mAnchorWidth[0]*mAnchorHeight[0]);
	ptrTestFarLeftRow = (float *)malloc(sizeof(float)*mAnchorWidth[0]*mAnchorHeight[0]);
	ptrTestCloseLeftRow = (float *)malloc(sizeof(float)*mAnchorWidth[0]*mAnchorHeight[0]);
	mptrEpipolarLength = (double *)malloc(sizeof(double)*mAnchorWidth[0]*mAnchorHeight[0]);

	ptrCoordiBlocks = (float *)malloc(sizeof(float)*maxDisparityG[0]*2*wG[0]*hG[0]);
	ptrOffsetBlocks = (char *)malloc(sizeof(char)*maxDisparityG[0]*4*wG[0]*hG[0]);
	for (int ii = 0; ii < maxDisparityG[0]*wG[0]*hG[0]; ++ii){
		ptrCoordiBlocks[ii*2] = _DEFAULT_ERROR_COORDI;
		ptrCoordiBlocks[ii*2 + 1] = _DEFAULT_ERROR_COORDI;
		ptrOffsetBlocks[ii*4] = _DEFAULT_ERROR_OFFSET;
		ptrOffsetBlocks[ii*4 + 1] = _DEFAULT_ERROR_OFFSET;
		ptrOffsetBlocks[ii*4 + 2] = _DEFAULT_ERROR_OFFSET;
		ptrOffsetBlocks[ii*4 + 3] = _DEFAULT_ERROR_OFFSET;
	}

	return 0;
}


int RenderMT(int start, int end, int lvl, uint8_t * ptrSrc, unsigned int * ptrBuffer){
	for (int iid = start; iid < end; ++iid){
		int targetIndex = iid*wG[lvl]*hG[lvl];
		uint8_t * ptrCurIntensity = (uint8_t * )(ptrBuffer + targetIndex);
		for (int vv = 0; vv < hG[lvl]; ++vv){
			for (int uu = 0; uu < wG[lvl]; ++uu){
				int destIdx = vv*wG[lvl] + uu;
				if (mptrdispscale[destIdx] == 0){
					ptrCurIntensity[vv*wG[lvl] + uu] = 0;
				}

				int coordiIdx = (vv*wG[lvl]*maxDisparityG[lvl] + uu*maxDisparityG[lvl] + iid)*2;
				float uu_t = ptrResCoordinate[coordiIdx];
				float vv_t = ptrResCoordinate[coordiIdx + 1];

				if (uu_t > 0){
					ptrCurIntensity[vv*wG[lvl] + uu] = int(getInterpolatedElementInt(ptrSrc, uu_t, vv_t, wG[lvl]) + 0.5);
				}
				else{
					ptrCurIntensity[vv*wG[lvl] + uu] = 0;
				}
			}
		}
	}

	return 0;
}

int CensusMT(int start, int end, int lvl, uint8_t * ptrMidBuffer, uint8_t * ptrDest){
	for (int iid = start; iid < end; ++iid){
		uint8_t * curDest = ptrDest + iid*wG[lvl]*hG[lvl]*4;
		memcpy(ptrMidBuffer, curDest, wG[lvl]*hG[lvl]);
		CensusTransformCircle_Test((uint32_t *)curDest, ptrMidBuffer, hG[lvl], wG[lvl], 4);
	}

	return 0;
}


int BuildMT(int start, int end, uint32_t * dest, uint32_t * left, uint32_t * right, int PatternSize, int lvl){

	int width = wG[lvl];
	int DisparityNum = maxDisparityG[0];
	int c1; //c2,c3,c4;
	int radius = PatternSize / 2;
	uint32_t invalid = 1 << 31;

	for (int iid = start; iid < end; ++iid){
		int row = radius + iid / (width - 2*radius);
		int col = radius + iid % (width - 2*radius);
		int pixelIndex = row*width + col;

		int scale = mptrdispscale[pixelIndex];
		if (scale == 0){
			continue;
		}
		int destRowindex = row*width*DisparityNum;
		int destColindex = col*DisparityNum;
		
		uint32_t leftvalue = left[pixelIndex];

		// printf("%d %d %d\n", row, col, scale);
		for(int d = 0; d < maxDisparityG[lvl]; d += scale){

			int rightIndex = wG[lvl]*hG[lvl]*d + pixelIndex;
			uint32_t rightValue = right[rightIndex];
			if (rightValue == invalid){
				continue;
			}

			c1 = leftvalue ^ rightValue;
			dest[destRowindex + destColindex + d] = c1;
		}
	}

	return 0;
}

int StereoMatch::RSProjection3(std::vector<double> motionState, int lvl){

	Vec3 Velocity(motionState[3] * RowTimeG, motionState[4] * RowTimeG, motionState[5] * RowTimeG);
	Vec3 Angular(motionState[0] * RowTimeG, motionState[1] * RowTimeG, motionState[2] * RowTimeG);

	// Reset buffer to zeros
	memcpy(ptrResCoordinate, ptrCoordiBlocks, sizeof(float)*maxDisparityG[lvl]*2*wG[lvl]*hG[lvl]);
	memcpy(ptrResCoordinateR2L, ptrCoordiBlocks, sizeof(float)*maxDisparityG[lvl]*2*wG[lvl]*hG[lvl]);
	memcpy(ptrResOffset, ptrOffsetBlocks, sizeof(char)*maxDisparityG[lvl]*4*wG[lvl]*hG[lvl]);

    bool multiThread = true;
    int threadCnt = THREAD_CNT;
    int taskCnt = mAnchorHeight[lvl]*mAnchorWidth[lvl];
    int thread_step = int(ceil(taskCnt/threadCnt));
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    if(multiThread){
		double debugInfo[256] = {0};

        std::vector<std::thread> thread_pool;
        for (int it = 0; it < threadCnt; ++it){
            int pt_start = thread_step * it;
            int pt_end = thread_step * (it + 1);

			if (pt_end > taskCnt){
				pt_end = taskCnt;
			}

            std::thread this_thread(RSProjectionSubset2RTGrid2, Angular, Velocity, pt_start, pt_end, debugInfo + 3*it, lvl);
            thread_pool.push_back(std::move(this_thread));
        }
        for (unsigned int it = 0; it < thread_pool.size(); ++it){
            if(thread_pool[it].joinable())
                thread_pool[it].join();
        }
    }
    else{
		double debugInfo[256] = {0};
        RSProjectionSubset2RTGrid2(Angular, Velocity, 0, taskCnt, debugInfo, lvl);
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack0 = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

	t1 = std::chrono::steady_clock::now();
	if (mGridSize[lvl] != 1){ // require interpolate
		taskCnt = (mAnchorHeight[lvl] - 1)*(mAnchorWidth[lvl] - 1);
		thread_step = int(ceil(taskCnt/threadCnt));
		if(multiThread){
			std::vector<std::thread> thread_pool;
			for (int it = 0; it < threadCnt; ++it){
				int pt_start = thread_step * it;
				int pt_end = thread_step * (it + 1);

				if (pt_end > taskCnt){
					pt_end = taskCnt;
				}

				std::thread this_thread(InterpolateProj2, pt_start, pt_end, lvl);
				thread_pool.push_back(std::move(this_thread));
			}
			for (unsigned int it = 0; it < thread_pool.size(); ++it){
				if(thread_pool[it].joinable())
					thread_pool[it].join();
			}
		}
		else{
			InterpolateProj2(0, taskCnt, lvl);
		}
	}

	bool IsShowRenderResult = true;
	if (IsShowRenderResult){
		taskCnt = maxDisparityG[lvl];
		thread_step = int(ceil(taskCnt/threadCnt));
		if(multiThread){
			std::vector<std::thread> thread_pool;
			for (int it = 0; it < threadCnt; ++it){
				int pt_start = thread_step * it;
				int pt_end = thread_step * (it + 1);

				if (pt_end > taskCnt){
					pt_end = taskCnt;
				}

				std::thread this_thread(RenderMT, pt_start, pt_end, lvl, mrightFrame->grayPyr[lvl], ptrCostVolumes);
				thread_pool.push_back(std::move(this_thread));
			}
			for (unsigned int it = 0; it < thread_pool.size(); ++it){
				if(thread_pool[it].joinable())
					thread_pool[it].join();
			}
		}
		else{
			RenderMT(0, taskCnt, lvl, mrightFrame->grayPyr[lvl], ptrCostVolumes);
		}
	}

	// Feature extraction
	bool IsTransferCensus = true;
	if (IsTransferCensus){
		taskCnt = maxDisparityG[lvl];
		thread_step = int(ceil(taskCnt/threadCnt));
		if(multiThread){
			std::vector<std::thread> thread_pool;
			for (int it = 0; it < threadCnt; ++it){
				int pt_start = thread_step * it;
				int pt_end = thread_step * (it + 1);

				if (pt_end > taskCnt){
					pt_end = taskCnt;
				}

				std::thread this_thread(CensusMT, pt_start, pt_end, lvl, ptrFinalCost + it*wG[lvl]*hG[lvl], (uint8_t *)ptrCostVolumes);
				thread_pool.push_back(std::move(this_thread));
			}
			for (unsigned int it = 0; it < thread_pool.size(); ++it){
				if(thread_pool[it].joinable())
					thread_pool[it].join();
			}
		}
		else{
			CensusMT(0, taskCnt, lvl, ptrFinalCost + 0*wG[lvl]*hG[lvl], (uint8_t *)ptrCostVolumes);
		}
	}


	// Build Cost Volumes
	bool IsBuildCostVolumes = true;
	if (IsBuildCostVolumes){
		memset(ptrCostVolumes2, 0xff, 4*wG[0]*hG[0]*m_disparity_size);
		int radius = m_pattern_size / 2; // 4
		taskCnt = (hG[lvl] - 2*radius)*(wG[lvl] - 2*radius);
		thread_step = int(ceil(taskCnt/threadCnt));
		if(multiThread){
			std::vector<std::thread> thread_pool;
			for (int it = 0; it < threadCnt; ++it){
				int pt_start = thread_step * it;
				int pt_end = thread_step * (it + 1);

				if (pt_end > taskCnt){
					pt_end = taskCnt;
				}

				// m_pattern_size
				std::thread this_thread(BuildMT, pt_start, pt_end, ptrCostVolumes2, ptrLeftCensus[lvl], ptrCostVolumes, m_pattern_size, lvl);
				thread_pool.push_back(std::move(this_thread));
			}
			for (unsigned int it = 0; it < thread_pool.size(); ++it){
				if(thread_pool[it].joinable())
					thread_pool[it].join();
			}
		}
		else{
			BuildMT(0, taskCnt, ptrCostVolumes2, ptrLeftCensus[lvl], ptrCostVolumes, m_pattern_size, lvl);
		}
	}

	// Interpolate thr projection from R to L
	if (mGridSize[lvl] != 1){ // require interpolate
		taskCnt = maxDisparityG[lvl];
		thread_step = int(ceil(taskCnt/threadCnt));
		if(multiThread){
			std::vector<std::thread> thread_pool;
			for (int it = 0; it < threadCnt; ++it){
				int pt_start = thread_step * it;
				int pt_end = thread_step * (it + 1);

				if (pt_end > taskCnt){
					pt_end = taskCnt;
				}

				std::thread this_thread(InterpolateProjR2L, pt_start, pt_end, lvl);
				thread_pool.push_back(std::move(this_thread));
			}
			for (unsigned int it = 0; it < thread_pool.size(); ++it){
				if(thread_pool[it].joinable())
					thread_pool[it].join();
			}
		}
		else{
			InterpolateProjR2L(0, taskCnt, lvl);
		}
	}

	bool IsShowPatchSizeChange = false;
	if (IsShowPatchSizeChange && lvl == 0){
		int step = 4;
		for (int ii = 0; ii < maxDisparityG[lvl]; ++ii){
			cv::Mat temp(hG[lvl], wG[lvl], CV_8U, cv::Scalar(0));
			float max_length = -1;
			float min_length = 10e8;
			for (int vv = step; vv < hG[lvl] - step; ++vv){
				for (int uu = step; uu < wG[lvl] - step; ++uu){
					int baseIndex00 = (vv*wG[lvl]*maxDisparityG[lvl] + uu*maxDisparityG[lvl] + ii)*2;
					int baseIndexR = (vv*wG[lvl]*maxDisparityG[lvl] + (uu+step)*maxDisparityG[lvl] + ii)*2;
					int baseIndexL = (vv*wG[lvl]*maxDisparityG[lvl] + (uu-step)*maxDisparityG[lvl] + ii)*2;
					int baseIndexU = ((vv-step)*wG[lvl]*maxDisparityG[lvl] + uu*maxDisparityG[lvl] + ii)*2;
					int baseIndexB = ((vv+step)*wG[lvl]*maxDisparityG[lvl] + uu*maxDisparityG[lvl] + ii)*2;
					if (ptrResCoordinate[baseIndex00] > 0 && 
						ptrResCoordinate[baseIndexR] > 0 &&
						ptrResCoordinate[baseIndexL] > 0 &&
						ptrResCoordinate[baseIndexU] > 0 &&
						ptrResCoordinate[baseIndexB] > 0){

						float SumDistance = 0;
						SumDistance += sqrtf((ptrResCoordinate[baseIndexR] - ptrResCoordinate[baseIndex00]) * 
										(ptrResCoordinate[baseIndexR] - ptrResCoordinate[baseIndex00]) + 
									(ptrResCoordinate[baseIndexR + 1] - ptrResCoordinate[baseIndex00 + 1]) * 
										(ptrResCoordinate[baseIndexR + 1] - ptrResCoordinate[baseIndex00 + 1]));
						SumDistance += sqrtf((ptrResCoordinate[baseIndexL] - ptrResCoordinate[baseIndex00]) * 
										(ptrResCoordinate[baseIndexL] - ptrResCoordinate[baseIndex00]) + 
									(ptrResCoordinate[baseIndexL + 1] - ptrResCoordinate[baseIndex00 + 1]) * 
										(ptrResCoordinate[baseIndexL + 1] - ptrResCoordinate[baseIndex00 + 1]));
						SumDistance += sqrtf((ptrResCoordinate[baseIndexU] - ptrResCoordinate[baseIndex00]) * 
										(ptrResCoordinate[baseIndexU] - ptrResCoordinate[baseIndex00]) + 
									(ptrResCoordinate[baseIndexU + 1] - ptrResCoordinate[baseIndex00 + 1]) * 
										(ptrResCoordinate[baseIndexU + 1] - ptrResCoordinate[baseIndex00 + 1]));
						SumDistance += sqrtf((ptrResCoordinate[baseIndexB] - ptrResCoordinate[baseIndex00]) * 
										(ptrResCoordinate[baseIndexB] - ptrResCoordinate[baseIndex00]) + 
									(ptrResCoordinate[baseIndexB + 1] - ptrResCoordinate[baseIndex00 + 1]) * 
										(ptrResCoordinate[baseIndexB + 1] - ptrResCoordinate[baseIndex00 + 1]));
						double targetValue = fabs(SumDistance - 4.0 * step) * 255;
						if (targetValue < 0) targetValue = 0;
						if (targetValue > 255) targetValue = 255;
						temp.at<uint8_t>(vv*wG[lvl] + uu) = targetValue;
						if (max_length < SumDistance) max_length = SumDistance;
						if (min_length > SumDistance) min_length = SumDistance;
					}
					else{
						temp.at<uint8_t>(vv*wG[lvl] + uu) = 0;
					}
				}
			}
			char buffer[1024] = {0};
			printf("lvl:%d disp:%d max:%f min:%f\n", lvl, ii, max_length, min_length);
			cv::imshow(buffer, temp);
			cv::waitKey(0);
		}
	}

    t2 = std::chrono::steady_clock::now();
    double ttrack1 = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    printf("RSProjection0: %fs RSProjection1 %fs\n", ttrack0, ttrack1);

    return 0;
}


// Extract feature for left and right pyramid
int StereoMatch::SetImagePair(FramePym * leftFrame, FramePym * rightFrame){
	mleftFrame = leftFrame;
	mrightFrame = rightFrame;
	for (int lvl = 0; lvl < PyrLevelsUsedG; ++lvl){ // PyrLevelsUsedG
		CensusTransformCircle(ptrLeftCensus[lvl], leftFrame->grayPyr[lvl], hG[lvl], wG[lvl], m_pattern_size, NULL);
	}
	return 0;
}

void StereoMatch::GenerateBaselineMap(cv::Mat & baselineMap){
	cv::Mat ratiomap(hG[0], wG[0], CV_64F, mptrBaselineAve);
	ratiomap.convertTo(baselineMap, CV_8U, 255. / 2);
}
}
