#include "common.h"
#include "dfr_syncfree.h"

extern "C" void test_solve_L_analysis_multirow(const char * filename, int * csrRowPtrL, int * csrColIdxL, VALUE_TYPE * csrValL_tmp, int n);
extern "C" void test_cusparse(const char * filename, int * csrRowPtrL, int * csrColIdxL, VALUE_TYPE * csrValL_tmp, int n);

extern "C" void test_solve_L_analysis_multirow_Franco(const char * filename, int * csrRowPtrL, int * csrColIdxL, VALUE_TYPE * csrValL_tmp, int n);
extern "C" void test_two_streams(const char * filename, int * csrRowPtrL, int * csrColIdxL, VALUE_TYPE * csrValL_tmp, int n);