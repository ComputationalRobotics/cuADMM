/***********************************************************************
* Compute Frobenius norm
***********************************************************************/

#include <math.h>
#include <mex.h>
#include <matrix.h>
#include <string.h> /* needed for memcpy() */
#include "header.h"



/**********************************************************
* 
***********************************************************/
void mexFunction(
      int nlhs,   mxArray  *plhs[], 
      int nrhs,   const mxArray  *prhs[] )

{        
    double   *X, *nrm;   
    double   nrm2=0.0;

    mwIndex  subs[2];
    mwSize   nsubs=2;
    mwSize   m, n, j, k, jm, isspX, kstart, kend; 
    mwIndex  *irX, *jcX;

/* CHECK FOR PROPER NUMBER OF ARGUMENTS */

   if (nrhs > 1){
      mexErrMsgTxt("mexFnorm: requires at most 1 input arguments."); }
   if (nlhs > 1){ 
      mexErrMsgTxt("mexFnorm: requires at most 1 output argument."); }   

/* CHECK THE DIMENSIONS */

    m = mxGetM(prhs[0]); 
    n = mxGetN(prhs[0]); 
    X = mxGetPr(prhs[0]);
    isspX = mxIsSparse(prhs[0]);
    if (isspX) {
       irX = mxGetIr(prhs[0]); 
       jcX = mxGetJc(prhs[0]);
    }
    /***** create return argument *****/    
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL); 
    nrm = mxGetPr(plhs[0]);  

    /***** Do the computations in a subroutine *****/  
    if (!isspX) { 
       for (j=0; j<n; j++) { 
           jm = j*m; 
           for (k=0; k<m; k++) { nrm2 += SQR(X[k+jm]); }
       }
    } else {
       for (j=0; j<n; j++) {
          kstart = jcX[j]; kend = jcX[j+1];
          for (k=kstart; k<kend; k++) { nrm2 += SQR(X[k]); }
       }
    }
    nrm[0] = sqrt(nrm2); 
    return;
}
/**********************************************************/
