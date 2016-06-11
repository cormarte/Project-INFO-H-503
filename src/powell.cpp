#include <vector>
#include "nrutil.h"

using namespace std;

// Definitions
#define ITMAX 200 
#define GOLD 1.618034f // Default ratio for magnification (minimum bracketing research)
#define GLIMIT 100.0f // Maximum magnification (minimum bracketing research)
#define CGOLD 0.381960f // Golden ratio
#define ZEPS 1.0e-10f // ZEPS avoids a zero value for tol if x is equal to zero
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d)
#define TOL 2.0e-4f // Tolerance for brent algorithm
#define TINY 1.0e-25f

// Global variables
float globalDirVector[3] = { 0.0f, 0.0f, 0.0f }; // Vector for the 1D function
float globalPoint[3] = { 0.0f, 0.0f, 0.0f }; // Current point (transform vector)
int globalN; // Dimension of search (dimension of transformation space)
float (*globalFunction)(float*); // Function to minimize


float function1D(float point){

	/* Evaluates function 'globalFunction' at the point 'point' moving along one direction from 'globalPoint'.
	The target direction is given by a 1xn vector 'globalDirVector'.
	'globalPoint' is a 1xn initial point moved to new point 'point'
	'globalFunction' is the function to minimize, here the mutual inforation computation.
	'globalPoint', 'globalDirVector', 'globalN' and 'globalFunction' are declared as globals. */

	float f = 0.0f;
	float movingPoint[3] = { 0.0f, 0.0f, 0.0f };

	for (int j = 0; j < globalN; j++){

		movingPoint[j] = globalPoint[j] + point*globalDirVector[j];
	}

	f = globalFunction(movingPoint);

	return f;
}

float brent(float lim1, float lim12, float lim2, float(*function)(float), float tol, float *xmin){

	/* Finds the minimum by parabolic fit of 3 points 'ax', 'bx', 'cx'.
	Range of 'ax', 'bx' and 'cx' are determined by the brackets method.
	'xmin' is the minimum value, 'function' is the function to minimize.
	Golden search is used to avoid oscillation from max to min of the function.
	'tol' is the tolerance of accuracy to find the minimum of 'function'

	6 points of the function are used: 'a', 'b', 'u', 'v', 'w' & 'x'
	'a' and 'b' are the limits within the minimum is found
	'x' is the current value function of the minimum
	'w' is the second current value function of the minimum
	'v' is the previous value function of 'w'
	'u' is the current point where function is evaluated

	3 criteria:
	- Parabolic interpolation is fit through 'x', 'w', 'v' (1)
	- Limits of interpolation are bewtween 'a' and 'b' (2)
	- New value of 'x' has to be a step < 1/2 of the previous step (to avoid oscillation) (3) */

	int iter = 0;
	float a = 0.0f, b = 0.0f, u = 0.0f, v = 0.0f, w = 0.0f, x = 0.0f;
	float fu = 0.0f, fv = 0.0f, fw = 0.0f, fx = 0.0f;
	float xm = 0.0f; // Midpoint between a and b
	float p = 0.0f, q = 0.0f, r = 0.0f;

	float tol1 = 0.0f, tol2 = 0.0f;
	float e = 0.0f, d = 0.0f, etemp = 0.0f; // Distance of the second to last step to compare with last step 

	a = (lim1 < lim2 ? lim1 : lim2); // a < b
	b = (lim1 > lim2 ? lim1 : lim2);

	x = w = v = lim12; // Initializations
	fw = fv = fx = (*function)(x); // Initializations

	for (iter = 1; iter <= ITMAX; iter++) {

		xm = 0.5f * (a + b);
		tol2 = 2.0f * (tol1 = tol * fabs(x) + ZEPS); // Typical position of min, distanced by 2 * x * tol

		if (fabs(x - xm) <= (tol2 - 0.5*(b - a))){ // Limit of search between a and b
			                                       // Min is evaluated only if it is distanced from previously other
												   // evaluated points at distance > tol
			*xmin = x;
			return fx;
		}

		if (fabs(e) > tol1) { // Construct a trial parabolic ﬁt

			r = (x - w) * (fx - fv);
			q = (x - v) * (fx - fw);
			p = (x - v) * q - (x - w) * r;  // Numerator of the parabolic interpolation formula
			q = 2.0f * (q - r);		// Denominator of the parabolic interpolation formula

			if (q > 0.0){

				p = -p;
			}

			q = fabs(q);
			etemp = e;
			e = d;

			if (fabs(p) >= fabs(0.5f * q * etemp) || p <= q * (a - x) || p >= q * (b - x)){ // Criteria 3 and 1

				d = CGOLD * (e = (x >= xm ? a - x : b - x)); // Golden section step permits convergence
			}

			else {

				d = p / q; //Takes the parabolic step
				u = x + d;

				if (u - a < tol2 || b - u < tol2){

					d = SIGN(tol1, xm - x);
				}
			}
		}

		else {

			d = CGOLD*(e = (x >= xm ? a - x : b - x)); // Golden section step permits convergence
		}

		u = (fabs(d) >= tol1 ? x + d : x + SIGN(tol1, d));

		fu = (*function)(u); // Function evaluation per iteration

		if (fu <= fx) {	 // x is no more a min

			if (u >= x) {

				a = x;
			}

			else {

				b = x;
			}

			SHFT(v, w, x, u);
			SHFT(fv, fw, fx, fu);
		}

		else { // x stays a potential min

			if (u < x) {

				a = u;
			}

			else {

				b = u;
			}

			if (fu <= fw || w == x) {

				v = w;
				w = u;
				fv = fw;
				fw = fu;
			}

			else if (fu <= fv || v == x || v == w) {

				v = u;
				fv = fu;
			}
		}
	}

	nrerror("Too many iterations in brent");
	*xmin = x; // Never get here
	return fx; // Returns the function  value of xmin
}

void mnbrak(float *lim1, float *lim2, float *lim12, float *flim1, float *flim2, float *flim12, float (*function)(float)){

	/* Limit search of the minimum within brackets. Given limits 'lim1' and 'lim2', finds the lower limit 'lim12'.
	Point 'u' is interpolated using a parabolic interpolation method or a default magnification to limit the range
	of 'lim1', 'lim2' and 'lim12'. 'function' computes the value of the mutual information in 1D only. */

	float ulim = 0.0f;
	float u = 0.0f; // u is a 4th point calculated by parabolic extrapolation of lim1, lim2 and lim12 or by default magnification
	float fu = 0.0f; 
	
	float r = 0.0f;
	float q = 0.0f;
	float tmp = 0.0f;

	*flim1 = (*function)(*lim1);
	*flim2 = (*function)(*lim2);

	if (*flim2 > *flim1) {

		SHFT(tmp, *lim1, *lim2, tmp); // Switches lim1 and lim2, research of lim12 from lim1 to lim2 by descent
		SHFT(tmp, *flim2, *flim1, tmp);
	}

	*lim12 = (*lim2) + GOLD * (*lim2 - *lim1); // Initializes lim12 by golden ratio (0.38 - 0.62) (golden section search)
	*flim12 = (*function)(*lim12);

	while (*flim2 > *flim12) {

		r = (*lim2 - *lim1) * (*flim2 - *flim12); // r and q are used for parabolic interpolation
		q = (*lim2 - *lim12) * (*flim2 - *flim1);
		u = (*lim2) - ((*lim2 - *lim1) * r - (*lim2 - *lim12 )* q) / (2.0f * SIGN(FMAX(fabs(r - q), TINY), r - q)); // Parabolic interpolation

		ulim = (*lim2) + GLIMIT * (*lim12 - *lim2); // Max step GLIMIT to find a parabolic point between lim2 and lim12

		if ((*lim2 - u) * (u - *lim12) > 0.0) { // Parabolic u is between lim12 and lim2 

			fu = (*function)(u); // function and thus the mutual information of point pcom moved to u

			if (fu < *flim12) { // Minimim is between lim12 and lim2 (1st condition)

				*lim1 = (*lim2); // New limit
				*lim2 = u;
				*flim1 = (*flim2);
				*flim2 = fu;
				return;
			}

			else if (fu > *flim2) { // Minimum is between lim1 and u

				*lim12 = u;
				*flim12 = fu;
				return;
			}

			u = (*lim12) + GOLD * (*lim12 - *lim2); // Parabolic fit no use, u is moved with regard to lim12 (default magnification)
			fu = (*function)(u);
		}

		else if ((*lim12 - u) * (u - ulim) > 0.0) { // Parabolic ﬁt u is between lim12 and its allowed limit

			fu = (*function)(u);

			if (fu < *flim12) {

				SHFT(*lim2, *lim12, u, *lim12 + GOLD * (*lim12 - *lim2));
				SHFT(*flim2, *flim12, fu, (*function)(u));
			}
		}

		else if ((u - ulim) * (ulim - *lim12) >= 0.0f) { // Parabolic u is at its maximum allowed limit

			u = ulim;
			fu = (*function)(u);
		}

		else {									// u trespasses ulimit, default magnification is used

			u = (*lim12) + GOLD * (*lim12 - *lim2);
			fu = (*function)(u);
		}

		SHFT(*lim1, *lim2, *lim12, u);				// Delete the oldest point
		SHFT(*flim1, *flim2, *flim12, fu);
	}
}

void linmin(float* value, float* point, float* dirVector, int n, float(*function)(float*)){
	
	/* Finds the minimum of the function 'function' starting from an initial point 'point' (initial transform) 
	and a direction vector 'dirVector'. 'point' is set such that 'function(point)' is minimum along the direction
	'dirVector' starting from the point 'point'. */

	// Limits for bracketing the minimum
	float lim1 = 0.0f;
	float lim2 = 0.0f;
	float lim12 = 0.0f; 
	float flim1 = 0.0f;
	float flim2 = 0.0f;
	float flim12 = 0.0f;

	float xmin = 0.0f; // Minimum to find

	globalN = n; // n is stored to a global variable to be used function1D

	for (int j = 0; j < n; j++) {

		globalPoint[j] = point[j]; // point is stored to a global variable to be used function1D
		globalDirVector[j] = dirVector[j]; // dirVector is stored to a global variable to be used function1D
	}

	globalFunction = function;

	lim1 = 0.0f; // Initialization for the brackets
	lim2 = 1.0f;

	mnbrak(&lim1, &lim2, &lim12, &flim1, &flim2, &flim12, function1D); // New values for the brackets are computed
	*value = brent(lim1, lim2, lim12, function1D, TOL, &xmin); // Brent algorithm to find the minimum of the function
															   // value is value function of the minimum
	for (int j = 0; j < n; j++) {

		dirVector[j] *= xmin;	// new direction
		point[j] += dirVector[j];	// new point
	}
}

void powell(float* value, float* point, const int n, const float tolerance, float (*function)(float*)) {

	/* Find the minimum 'value' and its argument 'point' of the function 'function' along its 'n' dimensions 
	with a tolerance 'tolerance' of value 'TOL' and a limited number of iterations 'ITMAX'.

	Extrapolated point 'extrapolatedPoint', previous point 'previousPoint' and their values 'extrapolatedValue' and 'previousValue'
	are used as criteria to decide whether the old directions are better than the new ones or not.
	
	Here, this function is used to minimize the negative mutual information with regard to the transformation (Tx, Ty, Rz). */

	vector < vector < float > > unitVectors; // nxn matrix of unit vectors

	for (int i = 0; i < n; i++) {

		vector < float > unitVector;

		for (int j = 0; j < n; j++) {
		
			if (i == j) {
			
				unitVector.push_back(1.0f);
			}

			else {

				unitVector.push_back(0.0f);
			}
		}

		unitVectors.push_back(unitVector);
	}

	float dirVector[3] = { 0.0f, 0.0f, 0.0f }; // Current direction vector

	float maxDecrease = 0.0f; // Maximum function decrease
	int argMaxDecrease = 0; // Index of the maximum function decrease
	float condition = 0.0f;

	float previousValue = 0.0f; // Previous value
	float previousPoint[3] = { 0.0f, 0.0f, 0.0f }; // Previous point

	float extrapolatedValue = 0.0f; // Previous function value and extrapolated function value
	float extrapolatedPoint[3] = { 0.0f, 0.0f, 0.0f };

	int iteration = 0;

	*value = (*function)(point); // Apply the first transform (0, 0, 0)

	for (int j = 0; j < n; j++){

		previousPoint[j] = point[j]; // Save the initial point
	}

	for (iteration = 1; ; ++iteration) {

		previousValue = (*value);	// Previous value
		maxDecrease = 0.0;
		argMaxDecrease = 0;	 // Index of the maximum increase vector
				
		for (int i = 0; i < n; i++) { // At each iteration, loop over all directions in the set

			for (int j = 0; j < n; j++){

				dirVector[j] = unitVectors[j][i]; // Direction vector copy
			}

			extrapolatedValue = (*value); // Value function of point which is extrapolated a bit further along the current direction

			linmin(value, point, dirVector, n, function); // Minimize along one direction 

			if (extrapolatedValue - *value > maxDecrease) { // Save the value of the largest decrease so far

				maxDecrease = extrapolatedValue - *value;
				argMaxDecrease = i;
			}
		}

		if (2.0 * (previousValue - (*value)) <= tolerance * (fabs(previousValue) + fabs(*value)) + TINY) { // Minimum is found

			return;
		}

		if (iteration == ITMAX) {

			nrerror("Powell exceeded maximum iterations");
		}

		for (int j = 0; j < n; j++) { // Construct the extrapolated point and the average direction is moved
			
			extrapolatedPoint[j] = 2.0f * point[j] - previousPoint[j]; // Save the old starting point
			dirVector[j] = point[j] - previousPoint[j];
			previousPoint[j] = point[j];
		}

		extrapolatedValue = (*function)(extrapolatedPoint); // Function value at extrapolated point

		if (extrapolatedValue < previousValue) {

			condition = 2.0f * (previousValue - 2.0f * (*value) + extrapolatedValue) * SQR(previousValue - (*value) - maxDecrease) - maxDecrease * SQR(previousValue - extrapolatedValue); // condition on largest decrease, to keep or not the new direction

			if (condition < 0.0f) {

				linmin(value, point, dirVector, n, function); // Move towards the minimum in the new direction and save the new direction

				for (int j = 0; j < n; j++) {

					unitVectors[j][argMaxDecrease] = unitVectors[j][n - 1]; // Update unitVectors with new dirVector found for the new minimum
					unitVectors[j][n - 1] = dirVector[j];
				}
			}
		}
	}
}