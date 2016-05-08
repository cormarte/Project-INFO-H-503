/*
 Copyright Ramtin Shams (hereafter referred to as 'the author'). All rights 
 reserved. **Citation required in derived works or publications** 
 
 NOTICE TO USER:   
 
 Users and possessors of this source code are hereby granted a nonexclusive, 
 royalty-free license to use this source code for non-commercial purposes only, 
 as long as the author is appropriately acknowledged by inclusion of this 
 notice in derived works and citation of appropriate publication(s) listed 
 at the end of this notice in any derived works or publications that use 
 or have benefited from this source code in its entirety or in part.
   
 
 THE AUTHOR MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 IMPLIED WARRANTY OF ANY KIND.  THE AUTHOR DISCLAIMS ALL WARRANTIES WITH 
 REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 OR PERFORMANCE OF THIS SOURCE CODE.  
 
 Relevant publication(s):
	@inproceedings{Shams_ICSPCS_2007,
		author        = "R. Shams and R. A. Kennedy",
		title         = "Efficient Histogram Algorithms for {NVIDIA} {CUDA} Compatible Devices",
		booktitle     = "Proc. Int. Conf. on Signal Processing and Communications Systems ({ICSPCS})",
		address       = "Gold Coast, Australia",
		month         = dec,
		year          = "2007",
		pages         = "418-422",
	}

	@inproceedings{Shams_DICTA_2007a,
		author        = "R. Shams and N. Barnes",
		title         = "Speeding up Mutual Information Computation Using {NVIDIA} {CUDA} Hardware",
		booktitle     = "Proc. Digital Image Computing: Techniques and Applications ({DICTA})",
		address       = "Adelaide, Australia",
		month         = dec,
		year          = "2007",
		pages         = "555-560",
		doi           = "10.1109/DICTA.2007.4426846",
	};
*/

// includes, system
#include <stdlib.h>
#include <tchar.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <driver_types.h>
#include "cuda_basics.h"
#include "cuda_hist.h"
#include "cuda_mi.h"

// includes, kernels
#include "gpu_mi.cu"

extern "C" float cudaMIa(float *src1, float *src2, int length, int xbins, int ybins, double &time, cudaHistOptions *p_options /*= NULL*/, bool device /*= false*/)
{
	dim3 grid, block;
	int size = length * sizeof(float);
	//Device memory pointers
	float *d_src1, *d_src2, *d_hist, *d_rows, *d_cols;
	time = 0;
	unsigned int hTimer;
    CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	TIMER_CREATE;

	TIMER_START;
	if (!device)
	{
		//Allocate data on the device
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src1, size));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src2, size));

		//Copy src data to device memory
		CUDA_SAFE_CALL(cudaMemcpy(d_src1, src1, size, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_src2, src2, size, cudaMemcpyHostToDevice));
	}
	else
	{
		d_src1 = src1; d_src2 = src2;
	}

    CUDA_SAFE_CALL(cudaThreadSynchronize());								
    CUT_SAFE_CALL(cutStartTimer(hTimer));								

	CUDA_SAFE_CALL(cudaMalloc((void**) &d_hist, xbins * ybins * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_rows, ybins * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemset(d_rows, 0, ybins * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_cols, xbins * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemset(d_cols, 0, xbins * sizeof(float)));
	TIMER_PRINT("Loading data", 0);

	cudaHist2Da(d_src1, d_src2, d_hist, length, xbins, ybins, p_options, true);					//No need to initialize d_hist, it will be done by cudaHist

	int totalSamples = length;
	//Caculate MI from the joint histogram
	// H(X,Y)
	float Hxy = cudaEntropy(d_hist, xbins * ybins, true) / totalSamples + logf((float)totalSamples);

	// H(X)
	cudaSumAlongCols(d_hist, d_cols, xbins, ybins, true);
	float Hx = cudaEntropy(d_cols, xbins, true) / totalSamples + logf((float)totalSamples);

	// H(Y)
	cudaSumAlongRows(d_hist, d_rows, xbins, ybins, true);
	float Hy = cudaEntropy(d_rows, ybins, true) / totalSamples + logf((float)totalSamples);

	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	time = cutGetTimerValue(hTimer);
	CUT_SAFE_CALL(cutDeleteTimer(hTimer));

	if (!device)
	{
		CUDA_SAFE_CALL(cudaFree(d_src1));
		CUDA_SAFE_CALL(cudaFree(d_src2));
	}

	CUDA_SAFE_CALL(cudaFree(d_hist));
	CUDA_SAFE_CALL(cudaFree(d_rows));
	CUDA_SAFE_CALL(cudaFree(d_cols));
	TIMER_DELETE;

	return Hx + Hy - Hxy;
}

extern "C" float cudaMIb(float *src1, float *src2, int length, int xbins, int ybins, double &time, cudaHistOptions *p_options /*= NULL*/, bool device /*= false*/)
{
	dim3 grid, block;
	int size = length * sizeof(float);
	//Device memory pointers
	float *d_src1, *d_src2, *d_hist, *d_rows, *d_cols;
	time = 0;
	unsigned int hTimer;
    CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	TIMER_CREATE;

	TIMER_START;
	if (!device)
	{
		//Allocate data on the device
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src1, size));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src2, size));

		//Copy src data to device memory
		CUDA_SAFE_CALL(cudaMemcpy(d_src1, src1, size, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_src2, src2, size, cudaMemcpyHostToDevice));
	}
	else
	{
		d_src1 = src1; d_src2 = src2;
	}

    CUDA_SAFE_CALL(cudaThreadSynchronize());								
    CUT_SAFE_CALL(cutStartTimer(hTimer));								

	CUDA_SAFE_CALL(cudaMalloc((void**) &d_hist, xbins * ybins * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_rows, ybins * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemset(d_rows, 0, ybins * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_cols, xbins * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemset(d_cols, 0, xbins * sizeof(float)));
	TIMER_PRINT("Loading data", 0);

	cudaHist2Db(d_src1, d_src2, d_hist, length, xbins, ybins, p_options, true);					//No need to initialize d_hist, it will be done by cudaHist

	int totalSamples = length;
	//Caculate MI from the joint histogram
	// H(X,Y)
	float Hxy = cudaEntropy(d_hist, xbins * ybins, true) / totalSamples + logf((float)totalSamples);

	// H(X)
	cudaSumAlongCols(d_hist, d_cols, xbins, ybins, true);
	float Hx = cudaEntropy(d_cols, xbins, true) / totalSamples + logf((float)totalSamples);

	// H(Y)
	cudaSumAlongRows(d_hist, d_rows, xbins, ybins, true);
	float Hy = cudaEntropy(d_rows, ybins, true) / totalSamples + logf((float)totalSamples);

	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	time = cutGetTimerValue(hTimer);
	CUT_SAFE_CALL(cutDeleteTimer(hTimer));

	if (!device)
	{
		CUDA_SAFE_CALL(cudaFree(d_src1));
		CUDA_SAFE_CALL(cudaFree(d_src2));
	}

	CUDA_SAFE_CALL(cudaFree(d_hist));
	CUDA_SAFE_CALL(cudaFree(d_rows));
	CUDA_SAFE_CALL(cudaFree(d_cols));
	TIMER_DELETE;

	return Hx + Hy - Hxy;
}

extern "C" float cudaMI_Approx(float *src1, float *src2, int length, int xbins, int ybins, double &time, cudaHistOptions *p_options /*= NULL*/, bool device /*= false*/)
{
	dim3 grid, block;
	int size = length * sizeof(float);
	//Device memory pointers
	float *d_src1, *d_src2, *d_hist, *d_rows, *d_cols;
	time = 0;
	unsigned int hTimer;
    CUT_SAFE_CALL(cutCreateTimer(&hTimer));
	TIMER_CREATE;

	TIMER_START;
	if (!device)
	{
		//Allocate data on the device
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src1, size));
		CUDA_SAFE_CALL(cudaMalloc((void**) &d_src2, size));

		//Copy src data to device memory
		CUDA_SAFE_CALL(cudaMemcpy(d_src1, src1, size, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_src2, src2, size, cudaMemcpyHostToDevice));
	}
	else
	{
		d_src1 = src1; d_src2 = src2;
	}

    CUDA_SAFE_CALL(cudaThreadSynchronize());								
    CUT_SAFE_CALL(cutStartTimer(hTimer));								

	CUDA_SAFE_CALL(cudaMalloc((void**) &d_hist, xbins * ybins * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_rows, ybins * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemset(d_rows, 0, ybins * sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_cols, xbins * sizeof(float)));
	CUDA_SAFE_CALL(cudaMemset(d_cols, 0, xbins * sizeof(float)));
	TIMER_PRINT("Loading data", 0);

	cudaHist2D_Approx(d_src1, d_src2, d_hist, length, xbins, ybins, p_options, true);					//No need to initialize d_hist, it will be done by cudaHist

	int totalSamples = (int) cudaSum(d_hist, xbins * ybins, true);
	//Caculate MI from the joint histogram
	// H(X,Y)
	float Hxy = cudaEntropy(d_hist, xbins * ybins, true) / totalSamples + logf((float)totalSamples);

	// H(X)
	cudaSumAlongCols(d_hist, d_cols, xbins, ybins, true);
	float Hx = cudaEntropy(d_cols, xbins, true) / totalSamples + logf((float)totalSamples);

	// H(Y)
	cudaSumAlongRows(d_hist, d_rows, xbins, ybins, true);
	float Hy = cudaEntropy(d_rows, ybins, true) / totalSamples + logf((float)totalSamples);

	if (!device)
	{
		CUDA_SAFE_CALL(cudaFree(d_src1));
		CUDA_SAFE_CALL(cudaFree(d_src2));
	}

	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL(cutStopTimer(hTimer));
	time = cutGetTimerValue(hTimer);
	CUT_SAFE_CALL(cutDeleteTimer(hTimer));

	CUDA_SAFE_CALL(cudaFree(d_hist));
	CUDA_SAFE_CALL(cudaFree(d_rows));
	CUDA_SAFE_CALL(cudaFree(d_cols));
	TIMER_DELETE;

	return Hx + Hy - Hxy;
}

extern "C" float cudaEntropy(float *src, int length, bool device /*= false*/)
{
	float *d_src, *d_dst;
	int size = length * sizeof(float);
	if (!device)
	{
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_src, size));
		//Copy src data to device memory
		CUDA_SAFE_CALL(cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice));
	}
	else
	{
		d_src = src;
	}
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_dst, size));

	cudaEntropyUnary(d_src, d_dst, length, true);
	float res = cudaSum(d_dst, length, true);

	if (!device)																				
		CUDA_SAFE_CALL(cudaFree(d_src));
	CUDA_SAFE_CALL(cudaFree(d_dst));

	return res;
}

CUDA_UNARY(EntropyUnary);
