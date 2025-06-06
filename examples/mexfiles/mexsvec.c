/***********************************************************************
* mexsvec.c : C mex file 
*
*   x = mexsvec(blk,A,isspx,type); 
* 
*   Input: A     = nxn matrix
*          isspx = 0, store x as a dense vector
*                  1, store x as a sparse vector.
*          type = 0, stack upper triangular part of A column-wise 
*                 1, stack lower triangular part of A row-wise 
*
* SDPNAL: 
* Copyright (c) 2008 by
* Xinyuan Zhao, Defeng Sun, and Kim-Chuan Toh 
***********************************************************************/

#include <math.h>
#include <mex.h>
#include <matrix.h>
#include "header.h"

/**********************************************************
* single block: stack upper triangular part of A column-wise 
* mwSizeo a column vector
**********************************************************/
void svec1(mwSize n, double r2, 
           double *A, mwIndex *irA, mwIndex *jcA, mwSize isspA, 
           double *B, mwIndex *irB, mwIndex *jcB, mwSize isspB) 

{  mwSize idx, i, j, jn, k, kstart, kend, idxj; 
   
   if (!isspB & !isspA) { 
      idx = 0; 
      for (j=0; j<n; j++) { 
          jn = j*n; 
          for (i=0; i<j; i++) { 
              B[idx] = A[i+jn]*r2; 
              idx++; } 
          B[idx] = A[j+jn]; 
          idx++; 
      }
   } else if (isspB & !isspA) { 
      idx = 0; 
      idxj = 0; 
      for (j=0; j<n; j++) { 
          jn = j*n; 
          idxj += j; 
          for (i=0; i<j; i++) { 
              irB[idx] = i+idxj;               
              B[idx]   = A[i+jn]*r2; 
              idx++; } 
          irB[idx] = j+idxj; 
          B[idx]   = A[j+jn]; 
          idx++; 
      }  
      jcB[1] = idx;  
   } else if (!isspB & isspA) {
      idx = 0; 
      for (j=0; j<n; j++) { 
          idx += j; 
          kstart = jcA[j]; kend = jcA[j+1]; 
          if (kstart < kend) { 
             for (k=kstart; k<kend; k++) { 
                i = irA[k]; 
                if (i >= j) { break; } 
                B[idx+i] = A[k]*r2; 
             }
             if (i == j) {  B[idx+i] = A[k]; }        
	  }
      }
   } else if (isspB & isspA) {
      idx = 0; 
      idxj = 0; 
      for (j=0; j<n; j++) { 
          idxj += j; 
          kstart = jcA[j];  kend = jcA[j+1];
          if (kstart < kend) { 
             for (k=kstart; k<kend; k++) { 
                i = irA[k]; 
                if (i >= j) {  break;  } 
                irB[idx] = i+idxj;              
                B[idx]   = A[k]*r2; 
                idx++; } 
             if ( i == j) { 
                irB[idx] = j+idxj; 
                B[idx]   = A[k]; 
                idx++;  }
          }  
      }
      jcB[1] = idx;  
   }
return;
}
/**********************************************************
* multiple sub-blocks: stack upper triangular part of A 
* column-wise mwSizeo a column vector
**********************************************************/
void svec2(mwSize n, mwSize numblk, mwIndex *cumblksize, mwIndex *blknnz, 
           double r2, 
           double *A, mwIndex *irA, mwIndex *jcA, mwSize isspA, 
           double *B, mwIndex *irB, mwIndex *jcB, mwSize isspB) 

{  mwSize idx, i, j, jn, l, jstart, jend, istart;
   mwSize rowidx, idxj, k, kstart, kend; 
   
   if (!isspB & !isspA) { 
      idx = 0; 
      jstart = 0; jend = 0; 
      for (l=0; l<numblk; l++) { 
  	  jend = cumblksize[l+1];   
          istart = jstart; 
          for (j=jstart; j<jend; j++) { 
              jn = j*n; 
              for (i=istart; i<j; i++) { 
                  B[idx] = A[i+jn]*r2; 
                  idx++; } 
              B[idx] = A[j+jn]; 
              idx++;    }
          jstart = jend; 
      }
   } else if (isspB & !isspA) { 
      idx = 0; 
      jstart = 0; jend = 0; 
      for (l=0; l<numblk; l++) { 
  	  jend = cumblksize[l+1];   
          istart = jstart;  
          idxj = 0; 
          for (j=jstart; j<jend; j++) { 
              jn = j*n; 
              idxj += j-jstart;
              rowidx = blknnz[l]-istart+idxj; 
              for (i=istart; i<j; i++) { 
                  irB[idx] = rowidx+i;               
                  B[idx]   = A[i+jn]*r2; 
                  idx++; } 
              irB[idx] = rowidx+j; 
              B[idx]   = A[j+jn]; 
              idx++; 
          }
          jstart = jend; 
      }  
      jcB[1] = idx;  
   } else if (!isspB & isspA) { 
      jstart = 0; jend = 0; 
      for (l=0; l<numblk; l++) { 
  	  jend = cumblksize[l+1];  
          istart = jstart;
          idx = blknnz[l]; 
          for (j=jstart; j<jend; j++) { 
              idx += j-jstart; 
              kstart = jcA[j]; kend = jcA[j+1]; 
              if (kstart < kend) { 
                 for (k=kstart; k<kend; k++) { 
                     i = irA[k]; 
                     if (i >= j) { break; } 
                     B[idx+i-istart] = A[k]*r2; 
                  }
                  if (i == j) {  B[idx+i-istart] = A[k]; }        
	      }
          }
          jstart = jend; 
      }  
   } else if (isspB & isspA) {
      idx = 0; 
      jstart = 0; jend = 0; 
      for (l=0; l<numblk; l++) { 
  	  jend = cumblksize[l+1];  
          istart = jstart;
          idxj = 0; 
          for (j=jstart; j<jend; j++) { 
              idxj += j-jstart;
              rowidx = blknnz[l]-istart+idxj;
              kstart = jcA[j];  kend = jcA[j+1];
              if (kstart < kend) { 
                 for (k=kstart; k<kend; k++) { 
                     i = irA[k]; 
                     if (i >= j) {  break;  } 
                     irB[idx] = rowidx+i;              
                     B[idx]   = A[k]*r2; 
                     idx++; } 
                 if (i == j) { 
                    irB[idx] = rowidx+j; 
                    B[idx]   = A[k]; 
                    idx++;  }
              }  
          }
          jstart = jend; 
      }
      jcB[1] = idx;  
   } 
return;
}
/**********************************************************
* single block: stack upper lowere part of A row-wise 
* mwSizeo a column vector
**********************************************************/
void svec3(mwSize n, double r2, 
           double *A, mwIndex *irA, mwIndex *jcA, mwSize isspA, 
           double *B, mwIndex *irB, mwIndex *jcB, mwSize isspB) 

{  mwSize idx, rowidx, i, j, jn, k, kstart, kend; 
   
   if (!isspB & !isspA) { 
      idx = 0; 
      for (i=0; i<n; i++) { 
          for (j=0; j<i; j++) { 
              B[idx] = A[i+j*n]*r2; 
              idx++; } 
          B[idx] = A[j+j*n]; 
          idx++; 
      }
   } else if (isspB & !isspA) { 
      idx = 0; 
      rowidx = 0; 
      for (i=0; i<n; i++) { 
          rowidx += i; 
          for (j=0; j<i; j++) { 
              irB[idx] = j+rowidx;               
              B[idx]   = A[i+j*n]*r2; 
              idx++; } 
          irB[idx] = j+rowidx; 
          B[idx]   = A[j+j*n]; 
          idx++; 
      }  
      jcB[1] = idx;  
   } else if (!isspB & isspA) {
      for (j=0; j<n; j++) { 
          kstart = jcA[j]; kend = jcA[j+1]; 
          for (k=kstart; k<kend; k++) { 
               i = irA[k]; 
               if (j < i) { 
 		  idx = i*(i+1)/2; 
		  B[j+idx] = A[k]*r2; }
               else if (j==i) {
		  idx = i*(i+1)/2; 
		  B[j+idx] = A[k]; }        
	  }
      }
   } else if (isspB & isspA) {
      idx = 0; 
      for (j=0; j<n; j++) { 
          kstart = jcA[j];  kend = jcA[j+1];
          for (k=kstart; k<kend; k++) { 
              i = irA[k]; 
              if (j < i) {                  
                 irB[idx] = j + i*(i+1)/2;              
                 B[idx] = A[k]*r2;   
                 idx++; }
              else if (j==i) {
                 irB[idx] = j + i*(i+1)/2; 
                 B[idx] = A[k]; 
                 idx++;  
              }
          } 
      }
      jcB[1] = idx;  
   }
return;
}
/**********************************************************
* multiple sub-blocks: stack lower triangular part of A 
* row-wise into a column vector
**********************************************************/
void svec4(mwSize n, mwSize numblk, mwIndex *cumblksize, mwIndex *blknnz, 
           double r2, 
           double *A, mwIndex *irA, mwIndex *jcA, mwSize isspA, 
           double *B, mwIndex *irB, mwIndex *jcB, mwSize isspB) 

{  mwSize idx, i, i1, j, l, jstart, jend, istart;
   mwSize rowidx, idx2, k, kstart, kend; 
   
   if (!isspB) { 
      for (l=0; l<numblk; l++) { 
	  jstart = cumblksize[l]; jend = cumblksize[l+1];  
          istart = jstart;
          idx = blknnz[l]; 
          for (j=jstart; j<jend; j++) { 
              kstart = jcA[j]; kend = jcA[j+1]; 
              idx2 = idx + j-jstart; 
              for (k=kstart; k<kend; k++) { 
                  i = irA[k]; 
                  if (j < i) {
		     i1 = i-istart; 
                     B[idx2+i1*(i1+1)/2] = A[k]*r2; }
                  else if (j==i) {
		     i1 = i-istart; 
                     B[idx2+i1*(i1+1)/2] = A[k]; }        
              }
          }
      }  
   } else {
      idx = 0; 
      for (l=0; l<numblk; l++) { 
	  jstart = cumblksize[l]; jend = cumblksize[l+1];  
          istart = jstart;
          for (j=jstart; j<jend; j++) {               
	      rowidx = blknnz[l] + (j-jstart); 
              kstart = jcA[j];  kend = jcA[j+1];
              for (k=kstart; k<kend; k++) { 
		   i = irA[k]; 
                   if (j < i) {
                      i1 = i-istart; 
		      irB[idx] = rowidx + i1*(i1+1)/2;              
                      B[idx]   = A[k]*r2; 
                      idx++; } 
                   else if (j==i) { 
                      i1 = i-istart; 
                      irB[idx] = rowidx + i1*(i1+1)/2; 
                      B[idx]   = A[k]; 
                      idx++; }
	      }
          }
      }
      jcB[1] = idx;  
   }
return;
}
/**********************************************************
* 
***********************************************************/
void mexFunction(
      int nlhs,   mxArray  *plhs[], 
      int nrhs,   const mxArray  *prhs[] )

{    mxArray  *blk_cell_pr;
     double   *A,  *B,  *blksize; 
     mwIndex  *irA, *jcA, *irB, *jcB;
     mwIndex  *cumblksize, *blknnz;
     mwSize   mblk, isspA, isspB;

     mwIndex  subs[2];
     mwSize   nsubs=2; 
     mwSize   m, n, n2, nsub, k, index, numblk, NZmax, type; 
     double   r2; 

/* CHECK FOR PROPER NUMBER OF ARGUMENTS */

   if (nrhs < 2){
      mexErrMsgTxt("mexsvec: requires at least 2 input arguments."); }
   if (nlhs > 1){ 
      mexErrMsgTxt("mexsvec: requires 1 output argument."); }

/* CHECK THE DIMENSIONS */

    mblk = mxGetM(prhs[0]); 
    if (mblk > 1) { 
       mexErrMsgTxt("mexsvec: blk can have only 1 row."); }
    m = mxGetM(prhs[1]); 
    n = mxGetN(prhs[1]); 
    if (m != n) { 
       mexErrMsgTxt("mexsvec: matrix must be square."); }

    subs[0] = 0; subs[1] = 1;
    index = mxCalcSingleSubscript(prhs[0],nsubs,subs); 
    blk_cell_pr = mxGetCell(prhs[0],index);
    numblk  = mxGetN(blk_cell_pr);
    blksize = mxGetPr(blk_cell_pr); 
    if (numblk == 1) { 
       n2 = n*(n+1)/2; 
    } else { 
       cumblksize = mxCalloc(numblk+1,sizeof(mwSize)); 
       blknnz = mxCalloc(numblk+1,sizeof(mwSize)); 
       cumblksize[0] = 0; blknnz[0] = 0; 
       n = 0; n2 = 0; 
       for (k=0; k<numblk; ++k) {
           nsub = (mwSize) blksize[k];
           n  += nsub; 
           n2 += nsub*(nsub+1)/2;  
           cumblksize[k+1] = n; 
           blknnz[k+1] = n2;  }
    }
    /***** assign pointers *****/
    A = mxGetPr(prhs[1]); 
    isspA = mxIsSparse(prhs[1]); 
    if (isspA) {  irA = mxGetIr(prhs[1]); 
                  jcA = mxGetJc(prhs[1]); 
                  NZmax = mxGetNzmax(prhs[1]);  }             
    else { NZmax = n2; }
    if ((numblk > 1) & (!isspA)) {
       mexErrMsgTxt("mexsvec: matrix must be sparse for numblk > 1"); }
    if (nrhs > 2) { 
       if (mxGetM(prhs[2])>1) { isspB = (mwSize)*mxGetPr(prhs[2]); }
       else if (NZmax < n2/2) { isspB = 1; }
       else                   { isspB = 0; } 
    } else {        
       if (NZmax < n2/2) { isspB = 1; }
       else              { isspB = 0; }
    } 
    if (nrhs > 3) { type = (mwSize)*mxGetPr(prhs[3]); } 
    else          { type = 0; } 
    /***** create return argument *****/
    if (isspB) {
       plhs[0] = mxCreateSparse(n2,1,NZmax,mxREAL); 
       B = mxGetPr(plhs[0]);
       irB = mxGetIr(plhs[0]); 
       jcB = mxGetJc(plhs[0]); 
       jcB[0] = 0; 
    } else {
       plhs[0] = mxCreateDoubleMatrix(n2,1,mxREAL); 
       B = mxGetPr(plhs[0]);  
    }   
    /***** Do the computations in a subroutine *****/
    r2 = sqrt(2); 
    if (type == 0) { 
       if (numblk == 1) { 
          svec1(n,r2,A,irA,jcA,isspA,B,irB,jcB,isspB);  
       } else {
          svec2(n,numblk,cumblksize,blknnz,r2,A,irA,jcA,isspA,B,irB,jcB,isspB); 
       }
    } else {
       if (numblk == 1) { 
          svec3(n,r2,A,irA,jcA,isspA,B,irB,jcB,isspB);  
       } else {
          svec4(n,numblk,cumblksize,blknnz,r2,A,irA,jcA,isspA,B,irB,jcB,isspB); 
       }
    }
    return;
 }
/**********************************************************/
