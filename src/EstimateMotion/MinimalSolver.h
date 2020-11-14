#ifndef MINIMAL_SOLVER_H
#define MINIMAL_SOLVER_H

#include <stdio.h>

namespace SFRSS
{
int PreProcess();
int rref(double * A, int m, int n, int threadID); 
int MinSolver(double * R11, double * ptRes, int threadID);
}

#endif