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
 
 Relevant publicationB(s):
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

/**
	\file
		mi_example.cpp
	\brief
		Demonstrates use of GPU-based mutual information (MI) computation methods
		in a simple C++ application.

	\remark
		This code has been optimized and tested on nVidia 8800 GTX. If used on a different
		card, the execution configuration (i.e. the number of threads and blocks) may 
		have to be modified from the default value for best performance.
*/
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* 
	Include NVidia Cuda Utility header file. Make sure $(CUDA_INC_PATH) and
	$(NVSDKCUDA_ROOT)/common/inc are the include path of the project.
*/
#include <cutil.h>
/*
	Incldue GPU implementation headers
*/
#include "..\cuda\cuda_basics.h"
#include "..\cuda\cuda_hist.h"
#include "..\cuda\cuda_mi.h"

#define BUFFER_LEN			10000000
#define	BINS				80

struct Options 
{
	int bins;
	unsigned int len;
	int threads, blocks;
};

Options ReadOptions(int argc, char *argv[])
{
	Options opts = {BINS, BUFFER_LEN, 0, 0};
	for (int i = 1; i < argc; i++)
	{
		if (strnicmp(argv[i], "-h", 2) == 0)
		{
			printf("Usage:\n");
			printf("mi_example [-b<number of bins>] [-l<data length>] [-k<number of GPU blocks>] [-t<threads per block>]\n");
			exit(0);
		}
		else if (strnicmp(argv[i], "-b", 2) == 0)
			opts.bins = strtol(argv[i] + 2, 0, 10); 
		else if (strnicmp(argv[i], "-l", 2) == 0)
			opts.len = strtol(argv[i] + 2, 0, 10); 
		else if (strnicmp(argv[i], "-t", 2) == 0)
			opts.threads = strtol(argv[i] + 2, 0, 10); 
		else if (strnicmp(argv[i], "-k", 2) == 0)
			opts.blocks = strtol(argv[i] + 2, 0, 10);
		else
		{
			printf("Invalid input argument. Use -h for help.");
			exit(1);
		}
	}

	return opts;
}

void cpuHist2D(float *src1, float *src2, unsigned int *hist, int length, int xbins, int ybins)
{
	for (int i = 0; i < length; i++)
	{
		unsigned int x = src1[i] * (xbins - 1) + 0.5f;
		unsigned int y = src2[i] * (ybins - 1) + 0.5f;
		hist[x + xbins * y]++;
	}
}

float cpuMI(float *src1, float *src2, int length, int xbins, int ybins, double &time)
{
	time = 0;
	unsigned int hTimer;
    CUT_SAFE_CALL(cutCreateTimer(&hTimer));

	CUT_SAFE_CALL(cutStartTimer(hTimer));								

	unsigned int *hist = new unsigned int [xbins * ybins];
	memset(hist, 0, sizeof(float) * xbins * ybins);

	cpuHist2D(src1, src2, hist, length, xbins, ybins);

	// Marginal and joint entropies
	float Hxy = 0, Hx = 0, Hy = 0;
	unsigned int *p_hist = hist;
	for (int y = 0; y < ybins; y++)
	{
		float Px = 0;
		for (int x = 0; x < xbins; x++, p_hist++)
		{
			if (*p_hist != 0)
			{
				float p = (float)*p_hist / length;
				Hxy -= p * log(p);
				Px += p;
			}
		}

		if (Px != 0)
			Hx -= Px * log(Px);
	}

	for (int x = 0; x < xbins; x++)
	{
		p_hist = hist + x;
		float Py = 0;
		for (int y = 0; y < ybins; y++, p_hist += xbins)
			Py += *p_hist;
		Py = Py / length;

		if (Py != 0)
			Hy -= Py * log(Py);
	}

	float MI = Hx + Hy - Hxy;
	delete []hist;

	CUT_SAFE_CALL(cutStopTimer(hTimer));
	time = cutGetTimerValue(hTimer);
	CUT_SAFE_CALL(cutDeleteTimer(hTimer));		

	return MI;
}

int main(int argc, char* argv[])
{
	// Read input options
	Options opt = ReadOptions(argc, argv);

	// Allocate two buffer for storage of randomly created input data
	float *buf1 = new float [opt.len];
	float *buf2 = new float [opt.len];

	// Change the random sequence everytime
	srand((unsigned)time(0));			

	// Initialize buffers with random values in [0,1] range
	// Note that the mutual information methods expect normalized data in [0,1]
	// range. The behavior of the methods is undefined if the input
	// contains out of range values.
	printf("Initializing the first random array of %d elements...\n", opt.len);
	for (unsigned int i = 0; i < opt.len; i++)
		buf1[i] = (int) ((float) rand() / RAND_MAX * 255) / 255.0f;
	
	printf("Initializing the second  array of %d elements...\n", opt.len);
	// Create a second array with some correlation to the first so that we get
	// some meaningful mutual information.
	for (unsigned int i = 0; i < opt.len - 1; i++)
	{
		buf2[i] = (buf1[i] + buf1[i+1]) / 2;
		buf2[i] = buf2[i] > 1.0f ? 1.0f : buf2[i];
	}
	buf2[opt.len - 1] = buf1[opt.len - 1];

	double t;
	float mi, mi_cpu;
	cudaHistOptions *p_opt = 0;
	if (opt.threads != 0)
	{
		p_opt = new cudaHistOptions;
		p_opt->blocks = opt.blocks;
		p_opt->threads = opt.threads;
	}

	mi_cpu = cpuMI(buf1, buf2, opt.len, opt.bins, opt.bins, t);
	printf("cpuMI (%dx%d bins): mi = %f, %.3f ms, %.1f MB/s\n", opt.bins, opt.bins, mi_cpu, t, opt.len * sizeof(float) / t*1e-3);

	mi = cudaMIa(buf1, buf2, opt.len, opt.bins, opt.bins, t, p_opt);
	printf("cudaMIa (%dx%d bins): mi = %f, %.3f ms, %.1f MB/s\n", opt.bins, opt.bins, mi, t, opt.len * sizeof(float) / t*1e-3);

	mi = cudaMIb(buf1, buf2, opt.len, opt.bins, opt.bins, t, p_opt);
	printf("cudaMIb (%dx%d bins): mi = %f, %.3f ms, %.1f MB/s\n", opt.bins, opt.bins, mi, t, opt.len * sizeof(float) / t*1e-3);

	mi = cudaMI_Approx(buf1, buf2, opt.len, opt.bins, opt.bins, t, p_opt);
	printf("cudaMI_Approx (%dx%d bins): mi = %f, %.3f ms, %.1f MB/s\n", opt.bins, opt.bins, mi, t, opt.len * sizeof(float) / t*1e-3);

	delete [] buf1;
	delete [] buf2;
	if (p_opt) delete p_opt;
	return 0;
}

