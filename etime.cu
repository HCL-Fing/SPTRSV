#include "common.h"

long int  t_ini = 0;
long int  t_ini_usec = 0;
int first_time = 1;

double current_time=0.0;

double eval_time()
{

	struct timeval time1;

	gettimeofday(&time1, NULL);

	// if (first_time){
	// 	first_time = 0;
	// 	t_ini = time1.tv_sec;
	// 	t_ini_usec = time1.tv_usec;
	// }

	double new_time = (double)time1.tv_sec * 1000.0 + (double)time1.tv_usec / 1000.0;
	double elapsed = new_time - current_time;
	current_time = new_time;

	return elapsed;
	// return  ((double)(time1.tv_sec) - (double)t_ini) * 1000.0 + (double) (time1.tv_usec - t_ini_usec) / 1000.0;
}
