/********************
#if !defined(MX_API_VER) || ( MX_API_VER < 0x07030000 )
typedef int mwIndex;
typedef int mwSize;
#endif
*********************/

#ifndef MWSIZE_MAX
    #define  mwIndex        int
    #define  mwSignedIndex  int
    #define  mwSize         int
#endif

#if !defined(_WIN32) && !defined(_WIN64)
#define dsyevd dsyevd_
#define dsyevx dsyevx_
#define dgesdd dgesdd_
#define dgemv dgemv_
#endif

#define mwSize  ptrdiff_t 
#define mwIndex ptrdiff_t 

#if !defined(SQR)
#define SQR(x) ((x)*(x))
#endif

#if !defined(MIN)
#define  MIN(A, B)   ((A) < (B) ? (A) : (B))
#endif

#if !defined(MAX)
#define  MAX(A, B)   ((A) > (B) ? (A) : (B))
#endif

