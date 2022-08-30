#ifndef __MATPROP__
#define __MATPROP__
#include <stdio.h>
#include <stdlib.h>


int validate_x( const VALUE_TYPE * x, int n, const char * func ){
    // validate x
    int err_counter = 0;

    for (int i = 0; i < n; i++)
    {
        //printf(" %f ", x[i]);
        if (abs(1 - x[i]) != 0 ){
            err_counter++;
            // printf("Error x[%i]=%f\n",i,x[i] );
        }
    }

    if (!err_counter){
        printf("\033[1;32m"); //Set the text to the color red
        printf("[PASS!]\n", func);
    }else{
        printf("\033[1;31m"); //Set the text to the color red
        printf("[FAILED! %d errors]\n", err_counter);
    }
    printf("\033[0m"); //Resets the text to default color
    return err_counter;
}


#endif
