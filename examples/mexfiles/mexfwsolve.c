/**********************************************************
* mexfwsolve: solve  R'*x = b where R is upper triangular
*            
*  x = mexfwsolve(Rin,b), where Rin = R
*
* SDPNAL: 
* Copyright (c) 2008 by
* Xinyuan Zhao, Defeng Sun, and Kim-Chuan Toh 
**********************************************************/

#include "mex.h"
#include <math.h>
#include <matrix.h>

#if !defined(MX_API_VER) || ( MX_API_VER < 0x07030000 )
typedef int mwIndex;
typedef int mwSize;
#endif

/********************************************************************
  PROCEDURE mexFunction - Entry for Matlab
*********************************************************************/
void mexFunction(const int nlhs, mxArray *plhs[],
                 const int nrhs, const mxArray *prhs[])
{
  double   *x, *b, *btmp, *R;
  mwIndex  *irb, *jcb, *irR, *jcR; 
  int      n, isspb;
  int      j, k, kstart, kend, idx; 
  double   tmp; 

  if (nrhs < 1) {
     mexErrMsgTxt("mexfwsolve: requires 2 input arguments."); }
  if (nlhs > 1) {
     mexErrMsgTxt("mexfwsolve: requires 1 output argument."); }

  n = mxGetM(prhs[1]); 
  isspb = mxIsSparse(prhs[1]);  
  if (isspb) {
     btmp = mxGetPr(prhs[1]);
     irb  = mxGetIr(prhs[1]); 
     jcb  = mxGetJc(prhs[1]); 
     b = mxCalloc(n,sizeof(double)); 
     kstart = jcb[0]; kend = jcb[1]; 
     for (k=kstart; k<kend; k++) { b[irb[k]] = btmp[k]; }    
  } else {
     b = mxGetPr(prhs[1]); 
  }
  if ( n != mxGetN(prhs[0]) ) { 
     mexErrMsgTxt("mexfwsolve: R should be square."); }
  if (!mxIsSparse(prhs[0])) {
     mexErrMsgTxt("mexfwsolve: R should be sparse."); }
  R   = mxGetPr(prhs[0]);
  irR = mxGetIr(prhs[0]); 
  jcR = mxGetJc(prhs[0]);

  if (irR[jcR[1]-1] > 0) {
     mexErrMsgTxt("mexfwsolve: R not upper triangular."); 
  }
  /*****************************/
  plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL);  
  x = mxGetPr(plhs[0]); 

  /************************************************
      x[j]*R[j,j] + x[0:j-1]*R[0:j-1,j] = b[j]
  ************************************************/
  x[0] = b[0]/R[0]; 
  for (j=1; j<n; j++) {
      kstart = jcR[j]; kend = jcR[j+1]-1; 
      tmp = 0.0; 
      for (k=kstart; k<kend; k++) { 
	  idx = irR[k]; 
          tmp += R[k]*x[idx];  	 
      }
      x[j] = (b[j]-tmp)/R[kend]; 
  }
  if (isspb) { mxFree(b); }
return;
}
/************************************************************************/




