/***********************************************************************
* mexeig.c : compute eigenvalue decomposition of a symmetric matrix
*            based on a divide-and-conquer method implemented in the
*            LAPACK routine dsyevd.f
* 
* Compilation in 
* unix: 
* mex -O -largeArrayDims -lmwlapack -lmwblas mexeig.c
*
* [V,D,flag] = mexeig(A); 
* compute both eigenvectors and eigenvalues;
* 
* [d,flag] = mexeig(A); 
* compute only eigenvalues.
*
* NNLS, version 0: 
* Copyright (c) 2009 by
* Kim-Chuan Toh and Sangwoon Yun 
***********************************************************************/

#include <math.h>
#include <mex.h>
#include <matrix.h>
#include <string.h> /* needed for memcpy() */
#include <blas.h>
#include <lapack.h>
#include "header.h"


/**********************************************************
* 
***********************************************************/
void mexFunction(
      int nlhs,   mxArray  *plhs[], 
      int nrhs,   const mxArray  *prhs[] )

{    double   *A, *V, *D, *work, *flag;  

     ptrdiff_t    subs[2];
     ptrdiff_t    nsubs=2;
     ptrdiff_t   *irD, *jcD, *work2;  
     ptrdiff_t    m, n, lwork, lwork2, info, j, k, jn; 
     char     *jobz="V";
     char     *uplo="U"; 

/* CHECK FOR PROPER NUMBER OF ARGUMENTS */

   if (nrhs > 1){
      mexErrMsgTxt("mexeig: requires at most 1 input argument."); }
   if (nlhs > 3){ 
      mexErrMsgTxt("mexeig: requires at most 3 output argument."); }   

/* CHECK THE DIMENSIONS */

    m = mxGetM(prhs[0]); 
    n = mxGetN(prhs[0]); 
    if (m != n) { 
       mexErrMsgTxt("mexeig: matrix must be square."); }
    if (mxIsSparse(prhs[0])) {
       mexErrMsgTxt("mexeig: sparse matrix not allowed."); }   
    A = mxGetPr(prhs[0]);     

    /***** create return argument *****/
    if (nlhs==3) { 
       plhs[0] = mxCreateDoubleMatrix(n,n,mxREAL); 
       V = mxGetPr(plhs[0]);  
       plhs[1] = mxCreateSparse(n,n,n,mxREAL); 
       D   = mxGetPr(plhs[1]); 
       irD = mxGetIr(plhs[1]); 
       jcD = mxGetJc(plhs[1]);
       plhs[2] = mxCreateDoubleMatrix(1,1,mxREAL); 
       flag = mxGetPr(plhs[2]);
       jobz="V";
    } else {
       plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL); 
       D = mxGetPr(plhs[0]); 
       V = mxCalloc(n*n,sizeof(double)); 
       plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL); 
       flag = mxGetPr(plhs[1]);      
       jobz="N";
    }
    /***** Do the computations in a subroutine *****/
    lwork  = 1+6*n+2*n*n;  
    work   = mxCalloc(lwork,sizeof(double)); 
    lwork2 = 3 + 5*n; 
    work2  = mxCalloc(lwork2,sizeof(ptrdiff_t)); 

    memcpy(V,A,(m*n)*sizeof(double)); 
    dsyevd(jobz,uplo,&n, V,&n, D, work,&lwork, work2,&lwork2, &info); 
    *flag = info; 
    
    if (nlhs==3) {
       for (k=0; k<n; k++) { irD[k] = k; }
       jcD[0] = 0;
       for (k=1; k<=n; k++) { jcD[k] = k; }  
    }
    return;
 }
/**********************************************************/
