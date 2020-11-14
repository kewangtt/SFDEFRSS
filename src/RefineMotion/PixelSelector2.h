
#pragma once
 
#include "../util/NumType.h"
#include "../util/FramePym.h"

namespace SFRSS
{

enum PixelSelectorStatus {PIXSEL_VOID=0, PIXSEL_1, PIXSEL_2, PIXSEL_3};


class FramePym;

class PixelSelector
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	int makeMaps(
			const FramePym* const fh,
			float* map_out, float density, int recursionsLeft=1, bool plot=true, float thFactor=1);

	PixelSelector(int w, int h);
	~PixelSelector();
	int currentPotential;


	bool allowFast;
	void makeHists(const FramePym* const fh);
private:

	Eigen::Vector3i select(const FramePym* const fh,
			float* map_out, int pot, float thFactor=1);


	unsigned char* randomPattern;


	int* gradHist;
	float* ths;
	float* thsSmoothed;
	int thsStep;
	const FramePym* gradHistFrame;
};




}

