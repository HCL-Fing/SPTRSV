#ifndef __MATPROP__
#define __MATPROP__


int get_nnz_max( int * csrRowPtrL,  int * csrColIdxL, int n ){

    int nnz_max = csrRowPtrL[1] - csrRowPtrL[0];

    for (int i = 1; i < n; ++i)
        if ( nnz_max < csrRowPtrL[i+1] - csrRowPtrL[i] ) nnz_max = csrRowPtrL[i+1] - csrRowPtrL[i];

    return nnz_max;
}

double get_nnz_stdev( int * csrRowPtrL,  int * csrColIdxL, int n ){


    int nnz = csrRowPtrL[n] - csrRowPtrL[0];
    double nnz_avg = (double) nnz / (double) n;

    double sum = 0;

    for (int i = 0; i < n; ++i)
        sum += pow((double)(csrRowPtrL[i+1] - csrRowPtrL[i]) - nnz_avg, 2);

    return sqrt( sum / (double) (n-1) ) ;
}

double get_avg_bw( int * csrRowPtrL,  int * csrColIdxL, int n ){

    double avg_bw = 1;

    for (int i = 1; i < n; ++i)
        avg_bw += (double)(i - csrColIdxL[csrRowPtrL[i]] + 1);

    return avg_bw/(double)n;
}

double get_locality_simple( int * csrRowPtrL,  int * csrColIdxL, int * inv_iorder, int wpb, int n ){

    double locality = 0;

    int nnz = csrRowPtrL[n] - csrRowPtrL[0];

    for (int i = 0; i < n; ++i)
    {
        for (int k = csrRowPtrL[i]; k < csrRowPtrL[i+1]; ++k)
        {
            if ( csrColIdxL[k]/wpb == i/wpb ) locality += 1.0 / (double) nnz;
        }
    }

    return locality;
}

double get_locality_multirow( int * csrRowPtrL,  int * csrColIdxL, int * inv_iorder, int wpb, int n ){

    double locality = 0;

    int nnz = csrRowPtrL[n] - csrRowPtrL[0];

    for (int i = 0; i < n; ++i)
    {
        for (int k = csrRowPtrL[i]; k < csrRowPtrL[i+1]; ++k)
        {
            if ( inv_iorder[csrColIdxL[k]]/wpb == inv_iorder[i]/wpb ) locality += 1.0 / (double) nnz;
        }
    }

    return locality;
}


int print_nnz( int * csrRowPtrL, int n ){

    FILE *ftabla;       
    ftabla = fopen("nnz_dist.dat","w");

	for (int i = 1; i <= n ; i++){
	    fprintf(ftabla,"%d\n", csrRowPtrL[i]-csrRowPtrL[i-1]);
	}

    fclose(ftabla);


}

int random_matrix(int* csrRowPtrL, int* csrColIdxL, int seed, int n, int salto){
	srand(seed);
	for(int f = 1; f < n; f++)
		for(int i = csrRowPtrL[f]; i < csrRowPtrL[f+1]-1; i++)
			if(i%salto == 0){
				if((i-1) < csrRowPtrL[f])		//Primero de la fila
					csrColIdxL[i] = csrColIdxL[i+1] -1 -rand()%max(1,csrColIdxL[i+1]);
				else	
					csrColIdxL[i] = csrColIdxL[i+1] -max(1, rand()%max(1,csrColIdxL[i+1] - csrColIdxL[i-1] ));
			}
	return seed;
}

int est_nnz_maxB(int * csrRowPtrL,  int first, int last, int chances){ //Last =n+1
    int mid =(last-first)/2 + first;
    //printf("Datos de la iteración \n %d %d %d \n", first, mid, last);
    if((last-first) <0)
        return -1;
    if (first == last){
        return 0;
    }
    if ((last-first) == 1)
        return csrRowPtrL[last] -csrRowPtrL[first];
    if(chances == 0){
        if((csrRowPtrL[last] - csrRowPtrL[mid]) < (csrRowPtrL[mid] - csrRowPtrL[first]))
            return est_nnz_maxB(csrRowPtrL, first, mid,0);
        else
            return est_nnz_maxB(csrRowPtrL,mid,last, 0);
    }else{
        int left= est_nnz_maxB(csrRowPtrL, first, mid, chances-1);
        int right= est_nnz_maxB(csrRowPtrL,mid,last, chances-1);
        if(left>right)
            return left;
        else
            return right;
    }

}


int est_nnz_max(int * csrRowPtrL,  int * csrColIdxL, int n, int partitions ){
    
    if (partitions>n)partitions=n;

    int nnz_max=0, rpp = n/partitions;
    for(int i = 1; i < partitions; i++){
        if(nnz_max < (csrRowPtrL[i*rpp] - csrRowPtrL[(i-1)*rpp])/ rpp)
            nnz_max = (csrRowPtrL[i*rpp] - csrRowPtrL[(i-1)*rpp])/ rpp;
    }
    //int desp = (n+1-(partitions-1)*rpp)/2,nnzC;
    if(nnz_max < ((csrRowPtrL[n] - csrRowPtrL[(partitions-1)*rpp])/ (n+1-(partitions-1)*rpp))) //La última tiene distinta cantidad de elementos
        nnz_max = (csrRowPtrL[n] - csrRowPtrL[(partitions-1)*rpp])/ (n+1-(partitions-1)*rpp);

/*  
	int nnz_max2
	int random = rand() % 25;
	rpp = (n+1)/(partitions-random);
	for(int i = 1; i < (partitions-random); i++){
        if(nnz_max2 < (csrRowPtrL[i*rpp] - csrRowPtrL[(i-1)*rpp])/ rpp)
            nnz_max2 = (csrRowPtrL[i*rpp] - csrRowPtrL[(i-1)*rpp])/ rpp;
    }

    if(nnz_max2 < ((csrRowPtrL[n] - csrRowPtrL[(partitions-random-1)*rpp])/ (n+1-(partitions-1-random)*rpp))) //La última tiene distinta cantidad de elementos
        nnz_max2 = (csrRowPtrL[n] - csrRowPtrL[(partitions-random-1)*rpp])/ (n+1-(partitions-1-random)*rpp);
    if(nnz_max2<nnz_max)
    	return nnz_max;
    else
    	return nnz_max2;





    /*if( (csrRowPtrL[n+1-desp]-csrRowPtrL[(partitions-1)*rpp])/((n+1-(partitions-1)*rpp)-desp) < ((csrRowPtrL[n+1]-csrRowPtrL[n+1-desp])/desp))
       nnzC = (csrRowPtrL[n+1]-csrRowPtrL[n+1-desp])/desp;
    else
        nnzC = (csrRowPtrL[n+1-desp]-csrRowPtrL[(partitions-1)*rpp])/((n+1-(partitions-1)*rpp)-desp);
    if(nnz_max > nnzC)
    */    return nnz_max;
    /*else
        return nnzC+1;
    */
}

int validate_x( const VALUE_TYPE * x, int n, const char * func ){
    // validate x
    int err_counter = 0;
    int first;
    for (int i = 0; i < n; i++)
    {
        //printf(" %f ", x[i]);
        if (abs(1 - x[i]) != 0 ){
            err_counter++;
            if(err_counter == 1) first=i;             
            // printf("Error x[%i]=%f\n",i,x[i] );
        }
    }

    if (!err_counter){
        printf("\033[1;32m"); //Set the text to the color red
        printf("[PASS!]\n", func);
    }else{
        printf("\033[1;31m"); //Set the text to the color red
        printf("[FAILED! %d errors] ", err_counter);
        printf("First error: %d\n",first);
    }
    printf("\033[0m"); //Resets the text to default color
    return err_counter;
}


#endif
