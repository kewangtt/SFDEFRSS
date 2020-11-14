#include "FramePym.h"
#include "globalCalib.h"
#include "FramePym.h"

namespace SFRSS
{
void FramePym::makeImages(uint8_t * color)
{
	// 填充 di, dip, 以及 absSquareGrad
	// di 和 dip 是3 channel， absSquareGrad 是单channel
	if (dIp[0] == NULL){
		for(int i = 0; i < PyrLevelsUsedG; i++){
			dIp[i] = new Eigen::Vector3f[wG[i]*hG[i]];
			absSquaredGrad[i] = new float[wG[i]*hG[i]];
			grayPyr[i] = new uint8_t[wG[i]*hG[i]];
		}
		dI = dIp[0];
		mColor = new float[wG[0]*hG[0]];
		grayX2 = new uint8_t[wG[0]*hG[0]*4];
	}

	// make d0
	int w = wG[0];
	int h = hG[0];
	for(int i = 0; i < w*h; i++){
		dI[i][0] = color[i];  // rgb have been transfer to grep map
		mColor[i] = color[i];
	}
	memcpy(grayPyr[0], color, wG[0]*hG[0]);

	// copy color map
	// memcpy((void *)mColor, (void *)color, w*h*sizeof(float));

	for(int lvl = 0; lvl < PyrLevelsUsedG; lvl++){
		int wl = wG[lvl], hl = hG[lvl];
		Eigen::Vector3f* dI_l = dIp[lvl];

		float* dabs_l = absSquaredGrad[lvl];
		if (lvl > 0){
			int lvlm1 = lvl-1;
			int wlm1 = wG[lvlm1];
			Eigen::Vector3f* dI_lm = dIp[lvlm1];

			for(int y = 0; y < hl; ++y)
				for(int x = 0; x < wl; ++x){
					dI_l[x + y*wl][0] = 0.25f * (dI_lm[2*x   + 2*y*wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1][0] +
												dI_lm[2*x   + 2*y*wlm1+wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1+wlm1][0]);

					grayPyr[lvl][x + y*wl] = (uint8_t)(dI_l[x + y*wl][0]);
				}
		}
		// Compute gradient and store in this array 
		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			float dx = 0.5f*(dI_l[idx+1][0] - dI_l[idx-1][0]);
			float dy = 0.5f*(dI_l[idx+wl][0] - dI_l[idx-wl][0]);


			if(!std::isfinite(dx)) dx=0;
			if(!std::isfinite(dy)) dy=0;

			dI_l[idx][1] = dx;
			dI_l[idx][2] = dy;

			dabs_l[idx] = dx*dx+dy*dy;
		}
	}
}

void FramePym::setX2Resolution(uint8_t * color){
	memcpy(grayX2, color, wG[0]*hG[0]*4);
}
}

