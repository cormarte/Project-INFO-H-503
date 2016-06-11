#define _USE_MATH_DEFINES

//#include <amp_math.h> // Elise
//#include <chrono>  //Elise
#include <math.h>
#include <iostream>
#include <vector>

#include "image.h"
#include "transform.h"
#include "nrutil.h"

//Elise
#include "stb_image.h"
#include "stb_image_write.h"

// Elise
//using namespace Concurrency::fast_math;
using namespace std;

// Definitions
#define HISTOGRAMSIZE 256
#define PI 3.14159265358979f
typedef int HistogramType;
const int histogramSize = 256;

//definition of variables (Powell)
#define ITMAX 200 
#define GOLD 1.618034 // default ration for magnification (min bracketing research)
#define GLIMIT 100.0 // maximum magnification (min bracketing research)
#define CGOLD 0.381960 //golden ratio
#define ZEPS 1.0e-10 // // ZEPS is a small number  to avoid to search for min which is
						// exactly at zero within the tolerance TOL
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d)
#define TOL 2.0e-4 // tolerance for brent algorithm
#define TINY 1.0e-25

// declaration of other variables (Powell)

int ncom; // dimension of search (dimension of transformation space) communicating with calculateFunc1D
float pcom[3] = {0.0f,0.0f,0.0f}; // point p (transform vector) communicating with calculateFunc1D
float dircom[3] = {0.0f,0.0f,0.0f}; // vector for one direction communicating with calculateFunc1D

// variables declared as global for MIfunc()
int fHeight, fWidth, fBitsPerPixel, rHeight, rWidth, rBitsPerPixel;

unsigned char* fPixels = stbi_load("..//data//floatingImage.jpg", &fWidth, &fHeight, &fBitsPerPixel, 1);
unsigned char* rPixels = stbi_load("..//data//referenceImage.jpg", &rWidth, &rHeight, &rBitsPerPixel, 1);

Image floatingImage = { fWidth, fHeight, fPixels };
Image referenceImage = { fWidth, rHeight, rPixels };
unsigned width = referenceImage.width;
unsigned height = referenceImage.height;

Image transformedImage = {width, height, new unsigned char[width * height]};

HistogramType* referenceHistogram = new HistogramType[HISTOGRAMSIZE];
HistogramType* transformedHistogram = new HistogramType[HISTOGRAMSIZE];
HistogramType* histogram2D = new HistogramType[HISTOGRAMSIZE * HISTOGRAMSIZE];


// Elise
float log2f( float n )  
{  
    // log(n)/log(2) is log2.  
    return log( n ) / log( (float) 2 );  
}


template<typename HistogramType, int histogramSize>
void cpuHistogram1D(const Image& image, HistogramType* histogram) {

	int height = image.height;
	int width = image.width;

	for (int y = 0; y != height; y++) {

		for (int x = 0; x != width; x++) {

			histogram[image.pixels[x + width*y]] += 1;
		}
	}
}

template<typename HistogramType, int histogramSize>
void cpuHistogram2D(const Image& image1, const Image& image2, HistogramType* histogram) {

	int height = image1.height;
	int width = image1.width;

	for (int y = 0; y != height; y++) {

		for (int x = 0; x != width; x++) {

			histogram[image1.pixels[x + width * y] + histogramSize * image2.pixels[x + width * y]] += 1;
		}
	}
}


template<typename HistogramType, int histogramSize>
float cpuMutualInformation(const HistogramType* histogram1, const HistogramType* histogram2, HistogramType* histogram2D, const unsigned int width, const unsigned int height){

	unsigned int histogramSum = width*height;

	float mutualInformation = 0;

	for (int bin1 = 0; bin1 != histogramSize; bin1++) {

		for (int bin2 = 0; bin2 != histogramSize; bin2++) {

			float p1 = (1.0f * histogram1[bin1]) / histogramSum;
			float p2 = (1.0f * histogram2[bin2]) / histogramSum;
			float p12 = (1.0f * histogram2D[bin1 + histogramSize * bin2]) / histogramSum;

			if (p12 != 0) {

				mutualInformation += p12*log2f(p12 / (p1 * p2));
			}
		}
	}

	return mutualInformation;
}

void cpuApplyTransform(const Image& originalImage, Image& transformedImage, const Transform& transform) {

	float cosrz = cosf(-transform.rz * PI / 180.0f);
	float sinrz = sinf(-transform.rz * PI / 180.0f);

	int centerX = originalImage.width / 2 - transform.tx;
	int centerY = originalImage.height / 2 - transform.ty;

	for (int y = 0; y != originalImage.height; y++) {

		for (int x = 0; x != originalImage.width; x++) {

			int originalX = (int)((x - centerX)*cosrz - (y - centerY)*sinrz - transform.tx + centerX);
			int originalY = (int)((x - centerX)*sinrz + (y - centerY)*cosrz - transform.ty + centerY);

			if (originalX >= 0 && (unsigned int)originalX < originalImage.width && originalY >= 0 && (unsigned int)originalY < originalImage.height) {

				transformedImage.pixels[x + transformedImage.width * y] = originalImage.pixels[originalX + originalImage.width * originalY];
			}

			else {

				transformedImage.pixels[x + transformedImage.width * y] = 255;
			}
		}
	}
}

float MIFunc(float transformVector[3]){
	/* calculates mutual information of reference image
	and transformed image with a transform vector transformVector*/
	
	Transform transform = { transformVector[0], transformVector[1], 0, 0, 0, transformVector[2]};
	
	cpuApplyTransform(floatingImage, transformedImage,transform);
	
	memset(histogram2D, 0, HISTOGRAMSIZE * HISTOGRAMSIZE * sizeof(HistogramType));
	cpuHistogram2D<HistogramType, HISTOGRAMSIZE>(transformedImage, referenceImage, histogram2D);
	
	// Because Powell search is written for minimization of function, MI is multiplied by -1 to search max
	float mutualInformation = (-1.0)*cpuMutualInformation<HistogramType, HISTOGRAMSIZE>(transformedHistogram, referenceHistogram, histogram2D, width, height);
	
	return mutualInformation;
}

float calculateFunc1D(float newP){

	/* Calculate function f on point newP which moved along one direction from pcom
	Direction is given by a 1xn vector direction dircom
	pcom is a 1xn initial point moved  to new point newP
	f is the function to minimize, here put as MIFunc
	pcom, dircom and ncom are declared as global
	*/

	int j=0;
	float f=0.0f;
	float movingP[3]={0.0f,0.0f,0.0f};

	for (j=0;j<ncom;j++){
		movingP[j]=pcom[j]+newP*dircom[j];
	}
	f=MIFunc(movingP);
	return f;
}

float brent(float lim1, float lim12, float lim2, float (*f)(float), float tol,float *xmin){
	/*calculates min by parabolic fit of 3 points ax,bx,cx
	interval ax,bx,cx are determined by brackets method
	xmin is the min result, f is the function to minimize
	Golden search is used to avoid oscillation from max to min of function
	tol is the tolerance of accuracy to find minimum

	6 points of the function are used: a,b,u,v,w,x
	a and b are the limits within the minimum is found
	x is the current value function of the min
	w is the second current value function of the min
	v is the previous value function of w
	u is the current point where function is evaluated

	3 criteria:
	- Parabolic interpolation is fit through x,w,v (1)
	- Limits of interpolation are bewtween a and b (2)
	- New value of x has to be a step < 1/2 of previous step (to avoid oscillation) (3)

	*/

	int iter = 0.0f;
	float a=0.0f,b=0.0f,u=0.0f,v=0.0f,w=0.0f,x=0.0f;
	float fu=0.0f,fv=0.0f,fw=0.0f,fx=0.0f;
	float xm=0.0f; // midpoint between a and b
	float p=0.0f,q=0.0f,r=0.0f;

	float tol1=0.0f,tol2=0.0f;
	float e=0.0f, d=0.0f,etemp=0.0f; // e = distance of step before last

	a=(lim1 < lim2 ? lim1 : lim2); // a < b
	b=(lim1 > lim2 ? lim1 : lim2);

	x=w=v=lim12; // initializations
	fw=fv=fx=(*f)(x); // initializations

	for (iter=1;iter<=ITMAX;iter++) {
		xm=0.5*(a+b);
		tol2=2.0*(tol1=tol*fabs(x)+ZEPS); // typical position of min, distanced by 2 * x * tol

		if (fabs(x-xm) <= (tol2-0.5*(b-a))){ // limit of search between a and b
											 // min is evaluated only if it is distanced from previously other
											 // evaluated points at distance > tol
			*xmin=x;
			return fx;
		}

		if (fabs(e) > tol1) { // Construct a trial parabolic ﬁt.
			r=(x-w)*(fx-fv);	
			q=(x-v)*(fx-fw);
			p=(x-v)*q-(x-w)*r;  // numerator of parabolic interpolation formula
			q=2.0*(q-r);		// denominator of  parabolic interpolation formula

			if (q > 0.0){
				p = -p;
			}

			q=fabs(q);
			etemp=e;
			e=d;
			if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x)){ //criterion 3  and 1
				d=CGOLD*(e=(x >= xm ? a-x : b-x));						// golden section step permits convergence
			}

			else {
				d=p/q;													//Take the parabolic step.
				u=x+d;
				if (u-a < tol2 || b-u < tol2){
					d=SIGN(tol1,xm-x);
				}
			}
		}
		else {
			d=CGOLD*(e=(x >= xm ? a-x : b-x));			// golden section step permits convergence
		}

		u=(fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));

		fu=(*f)(u);										// This is the one function evaluation per iteration.
		
		if (fu <= fx) {	 // x is no more a min
			if (u >= x){
				a=x;
			}
			else{
				b=x;
			}
			SHFT(v,w,x,u);	
			SHFT(fv,fw,fx,fu);
		}
		else { // x stays a potential min
			if (u < x) {
				a=u;
			}
			else {
				b=u;
			}
			if (fu <= fw || w == x) {
				v=w;
				w=u;
				fv=fw;
				fw=fu;
			}
			else if (fu <= fv || v == x || v == w) {
				v=u;
				fv=fu;
			}
		} 
	}
	nrerror("Too many iterations in brent");
	*xmin=x; //Never get here.
	return fx; // return value function of xmin
} 

void mnbrak(float *lim1, float *lim2, float *lim12, float *flim1, float *flim2, float *flim12,float (*func)(float)){ 
	
	/* Limit search of min between brackets.
	Given limits lim1 and lim2, it finds lim12 which is lower than lim1 and lim2.
	Point u in intrapolated by parabolic intrapolation or by default magnification 
	to limit the interval of lim1, lim2 and lim12.
	func is calculateFunc1D which calculates MI. */

	float ulim = 0.0f, u=0.0f, fu=0.0f; // u is a 4th point calculated by parabolic extrapolation of lim1,lim2 and lim12
										// or by default magnification
	float r=0.0f,q=0.0f,tmp=0.0f;
	*flim1=(*func)(*lim1);
	*flim2=(*func)(*lim2);

	if (*flim2 > *flim1) {  
		SHFT(tmp,*lim1,*lim2,tmp); // switch lim1 and lim2, research of lim12 from lim1 to lim2 by descent
		SHFT(tmp,*flim2,*flim1,tmp);
	}

	*lim12=(*lim2)+GOLD*(*lim2-*lim1); // initialize lim12 by golden ratio (0.38 - 0.62) (golden section search)
	*flim12=(*func)(*lim12);

	while (*flim2 > *flim12) {
		r=(*lim2-*lim1)*(*flim2-*flim12); // r and q for parabolic interpolation formula
		q=(*lim2-*lim12)*(*flim2-*flim1);
		u=(*lim2)-((*lim2-*lim1)*r-(*lim2-*lim12)*q)/(2.0*SIGN(FMAX(fabs(r-q),TINY),r-q)); // formula for parabolic interpolation
		
		ulim=(*lim2)+GLIMIT*(*lim12-*lim2); // max step GLIMIT to find a parabolic point between lim2 and lim12
		
		if ((*lim2-u)*(u-*lim12) > 0.0) { //Parabolic u is between lim12 and lim2 

			fu=(*func)(u);			// CalculateFunc1D and thus MI of point pcom moved to u

			if (fu < *flim12) {		// Minimim is  between lim12 and lim2 (1st condition)
				*lim1=(*lim2);		// New limit
				*lim2=u;
				*flim1=(*flim2);
				*flim2=fu;
			return;
			}

			else if (fu > *flim2) { // Minimum is between lim1 and u
				*lim12=u;
				*flim12=fu;
			return;
			}

			u=(*lim12)+GOLD*(*lim12-*lim2); // parabolic fit no use, u is moved in relation with lim12 (default magnification)
			fu=(*func)(u);
		}

		else if ((*lim12-u)*(u-ulim) > 0.0) { // Parabolic ﬁt u is between lim12 and its allowed limit
			fu=(*func)(u);
			if (fu < *flim12) {
				SHFT(*lim2,*lim12,u,*lim12+GOLD*(*lim12-*lim2));
				SHFT(*flim2,*flim12,fu,(*func)(u));
			}
		}
		else if ((u-ulim)*(ulim-*lim12) >= 0.0) { // Parabolic u is at maximum allowed limit
			u=ulim;
			fu=(*func)(u);
		}
		else {									// u trespasses ulimit, default magnification is used
			u=(*lim12)+GOLD*(*lim12-*lim2);
			fu=(*func)(u);
		}
		SHFT(*lim1,*lim2,*lim12,u);				// Eliminates oldest point
		SHFT(*flim1,*flim2,*flim12,fu);
	}
}

void linmin(float p[3], float dirVector[3], int n, float* fmin, float (*func)(float [3])){
	/* Find minimum of func starting from a initial point p (initial transform) and a direction vector dirVector
	p is reset to where fmin = func(p) taskes a minimum along the direction xi from p
	*/

	int j = 0;
	float lim1 = 0.0f, lim2 = 0.0f, lim12 = 0.0f; // limits for bracketing the min
	float flim1=0.0f,flim2=0.0f,flim12=0.0f;
	
	float xmin= 0.0f; // min to find

	ncom=n; // n, number or dimensions, to pass to calculateFunc1D

	for (j=0;j<n;j++) {
		pcom[j]=p[j]; // pcom, copy of initial point (initial transform), to pass to calculateFunc1D
		dircom[j]=dirVector[j]; // dircom, copy of a direction vector (one dimension) , to pass to calculateFunc1D
	}

	lim1=0.0; // Initialization for brackets
	lim2=1.0;

	mnbrak(&lim1,&lim2,&lim12,&flim1,&flim2,&flim12,calculateFunc1D); // new values for brackets
	*fmin =brent(lim1,lim2,lim12,calculateFunc1D,TOL,&xmin); // Brent algorithm to find min of function
													 // fmin is value function of min
	for (j=0;j<n;j++) {
		dirVector[j] *= xmin;	// new direction
		p[j] += dirVector[j];	// new point
		//cout << "p:" << p[j] << endl;
	}
}

void Powell(float p[3], float ** unitVector,int n, float ftol, int *iter, float* fmin,float(*func) (float [3])){

	/*Minimize of function func (MI) by research of the best transform in 3 dimensions (Tx, Ty, Rz).
	p is a initial transform and unitVector ix a nxn unit matrix. min is returned as p and is found with
	a tolerance ftol TOL with a limited number of iterations ITMAX. Value function of min is fmin
	*/

	int i = 0,ibig = 0,j= 0; 
	float del = 0.0f,t = 0.0f;
	float f0 = 0.0f;
	float fextrapolated= 0.0f;
	float p0[3] = {0.0f,0.0f,0.0f} ,pextrapolated[3] = {0.0f,0.0f,0.0f}; // copy of transform vector

	float dirVector[3] = {0.0f,0.0f,0.0f}; // copy for direction vectors

	*fmin=(*func)(p); // first transform (put at (0,0,0))
	
	for (j=0;j<n;j++){
		p0[j]=p[j]; //Save the initial point.
	}
	for (*iter=1;;++(*iter)) {

		//cout << "iter:" << *iter << endl;
		f0 = (*fmin);	//initial function value
		ibig=0;				// indices for upadtes of direction vector
		del=0.0;			// biggest function decrease

		for (i=0;i<n;i++) { //In each iteration, loop over all directions in the set.

			for (j=0;j<n;j++){
				dirVector[j]=unitVector[j][i]; //Copy one direction
			}

			fextrapolated=(*fmin); // value function of point which is extrapolated a bit further along the current direction

			linmin(p,dirVector,n,fmin,func); //minimize for one direction 

			if (fextrapolated-*fmin > del) { // and record it if it is the largest decrease so far.
				del=fextrapolated-*fmin;
				ibig=i;
			}
		}

		if (2.0*(f0-(*fmin)) <= ftol*(fabs(f0)+fabs(*fmin))+TINY) { // min is found
			return;
		}

		if (*iter == ITMAX) {
			nrerror("powell exceeding maximum iterations.");
		}

		for (j=0;j<n;j++) { //Construct the extrapolated point and the average direction moved. 
								//Save the old starting point.
			pextrapolated[j]=2.0*p[j]-p0[j];
			dirVector[j]=p[j]-p0[j];
			p0[j]=p[j];
		}

		fextrapolated=(*func)(pextrapolated); //Function value at extrapolated point.

		if (fextrapolated < f0) {
			t=2.0*(f0-2.0*(*fmin)+fextrapolated)*SQR(f0-(*fmin)-del)-del*SQR(f0-fextrapolated); // condition on largest decrease, to keep or not the new direction
			
			if (t < 0.0) {
				linmin(p,dirVector,n,fmin,func); // Move to the minimum of the new direction, and save the new direction.
				
				for (j=0;j<n;j++) {
					unitVector[j][ibig]=unitVector[j][n-1]; // Update unitVector with new dirVector found for new min
					unitVector[j][n-1]=dirVector[j];
				}
			}
		}
	} 
	
}

Image PowellRegister(const Image& floatingImage, const Image& referenceImage) {
	
	memset(transformedHistogram, 0, HISTOGRAMSIZE * sizeof(HistogramType));
	cpuHistogram1D<HistogramType, HISTOGRAMSIZE>(floatingImage, transformedHistogram);

	memset(referenceHistogram, 0, HISTOGRAMSIZE * sizeof(HistogramType));
	cpuHistogram1D<HistogramType, HISTOGRAMSIZE>(referenceImage, referenceHistogram);


	float transformVector[3] = {0.0f,0.0f,0.0f};
	float **unitVector;
	unitVector = new float *[3];

	for(int i = 0; i <3; i++)
		unitVector[i] = new float[3]();
	unitVector[0][0] = 1.0f;
	unitVector[1][1] = 1.0f;
	unitVector[2][2] = 1.0f;

	float* fmin = new float(0);
	int *iter = new int(0);

	Powell(transformVector, unitVector, 3, TOL, iter, fmin, MIFunc);

	Transform transform = { transformVector[0], transformVector[1], 0, 0, 0, transformVector[2] };
	cpuApplyTransform(floatingImage, transformedImage,transform);
	//HistogramType* histogram2D = cpuHistogram2D<HistogramType, HISTOGRAMSIZE>(transformedImage, referenceImage);
	//double mutualInformation = cpuMutualInformation<HistogramType, HISTOGRAMSIZE>(histogram2D);
	cout << "Optimal transform: Txbis: " << transform.tx << ", Tybis: " << transform.ty << ", Rzbis: " << transform.rz << endl;
	
	return transformedImage;

}
