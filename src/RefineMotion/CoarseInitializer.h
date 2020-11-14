

#pragma once

#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "vector"
#include "PixelSelector2.h"
#include "MatrixAccumulators.h"
#include "../util/settings.h"
#include "../util/NumType.h"
#include "../util/FramePym.h"

using namespace cv;

namespace SFRSS
{

struct Pnt
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	double u,v;
	double motion_u,motion_v;

	double idepth_candidates[4];
	int valid_idepth_num;

	double idepth_initial;
	double idepth;
	bool isGood;
	Vec2f energy;		// (UenergyPhotometric, energyRegularizer)
	bool isGood_new;
	double idepth_new;
	Vec2f energy_new;

	double ori_idepth;

	double lastHessian;
	double lastHessian_new;

	// max stepsize for idepth (corresponding to max. movement in pixel-space).
	double maxstep;

	int parent;
	double parentDist;

	int neighbours[10];
	double neighboursDist[10];

	double my_type;
	double outlierTH;

	double guessRightRow;
	double deltaRow;
};

class CoarseInitializer {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	CoarseInitializer(int w, int h);
	~CoarseInitializer();

	bool printDebug;

	Pnt* points[PYR_LEVELS];
	int numPoints[PYR_LEVELS];

	
	// New
	std::vector<double> Optimize3_withdepths_MT(std::vector<double> motionState, int pyramidLvl, bool FixTranslation);

	void setLeftFramePym(FramePym* newFrameHessian);
	void setRightFramePym(FramePym* newFrameHessian);
	// Create a new pixel selector...
	PixelSelector * sel;
	float densities[PYR_LEVELS];
	double m_row_time;

	float * statusMap;
	bool * statusMapB;

private:

	Eigen::DiagonalMatrix<float, 6> wM;

	Accumulator9 acc9;
	Accumulator9 acc9SC;

	void doStep(int lvl, float lambda, Vec6f inc);
	void applyStep(int lvl);

	void makeNN();

};




struct FLANNPointcloud
{
    inline FLANNPointcloud() {num=0; points=0;}
    inline FLANNPointcloud(int n, Pnt* p) :  num(n), points(p) {}
	int num;
	Pnt* points;
	inline size_t kdtree_get_point_count() const { return num; }
	inline float kdtree_distance(const float *p1, const size_t idx_p2,size_t /*size*/) const
	{
		const float d0=p1[0]-points[idx_p2].u;
		const float d1=p1[1]-points[idx_p2].v;
		return d0*d0+d1*d1;
	}

	inline float kdtree_get_pt(const size_t idx, int dim) const
	{
		if (dim==0) return points[idx].u;
		else return points[idx].v;
	}
	template <class BBOX>
		bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

}


