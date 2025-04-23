/**********************************************************
* mexbwsolve: solve  R*x = b where R is upper triangular
*            
*  x = mexbwsolve(Rin,b), where Rin = transpose(R). 
*
* SDPNAL: 
* Copyright (c) 2008 by
* Xinyuan Zhao, Defeng Sun, and Kim-Chuan Toh 
**********************************************************/

#include <mex.h>
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
  double   *x, *b, *btmp, *Rt;
  mwIndex  *irb, *jcb, *irRt, *jcRt; 
  int      n, isspb;
  int      j, k, kstart, kend, idx; 
  double   tmp; 

  if(nrhs < 1)
    mexErrMsgTxt("mexbwsolve: requires 2 input arguments.");
  if(nlhs > 1)
    mexErrMsgTxt("mexbwsolve: requires 1 output argument.");

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
     mexErrMsgTxt("mexbwsolve: Rt should be square."); }
  if (!mxIsSparse(prhs[0])) {
     mexErrMsgTxt("mexbwsolve: Rt should be sparse."); }
  Rt   = mxGetPr(prhs[0]);
  irRt = mxGetIr(prhs[0]); 
  jcRt = mxGetJc(prhs[0]);

  if (irRt[jcRt[n]-1] < n-1) { 
      mexErrMsgTxt("mexbwsolve: Rt not lower triangular.");
  } 
  /*****************************/
  plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL);  
  x = mxGetPr(plhs[0]); 

  /************************************************
      x[j]*Rt[j,j] + x[j+1:n]*Rt[j+1:n,j] = b[j]
  ************************************************/
  x[n-1] = b[n-1]/Rt[jcRt[n]-1];

  for (j=n-1; j>0; j--) {
      kstart = jcRt[j-1]+1; kend = jcRt[j]; 
      tmp = 0.0; 
      for (k=kstart; k<kend; k++) { 
	  idx = irRt[k]; 
          tmp += Rt[k]*x[idx];  	 
      }
      x[j-1] = (b[j-1]-tmp)/Rt[kstart-1]; 
  }
  if (isspb) { mxFree(b); }
return;
}
/************************************************************************/

