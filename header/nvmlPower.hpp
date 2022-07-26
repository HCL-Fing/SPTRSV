
#ifndef __INCLNVML__
#define __INCLNVML__
/*
Header file including necessary nvml headers.
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <nvml.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>

void nvmlAPIRun(const char *);
float nvmlAPIEnd();
void *powerPollingFunc(void *ptr);
int getNVMLError(nvmlReturn_t resultToCheck);

#endif
