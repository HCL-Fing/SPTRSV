#include "common.h"
#include "mmio.h"
#include "transpose.h"

int min(int a, int b){
    return (a<b)?a:b;
}
int max(int a, int b){
	return (a>b)?a:b;
}


// #include "spts_syncfree_serialref.h"
// #include "spts_syncfree_cuda.h"
// #include "dfr_syncfree.h"


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
void prueba(int* csrRowPtrL, VALUE_TYPE* csrValL){
	for(int i=1; i<4;i++){
		for(int j = csrRowPtrL[i-1]+1; j<=csrRowPtrL[i];j++)
			printf("F:%i V:%f\n", i, csrValL[j]);
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

double est_nnz_stdev(int* csrRowPtrL, int n, int partitions){

    if (partitions>n)partitions=n;

    int rpp = n/partitions;

    int nnz = csrRowPtrL[n] -csrRowPtrL[0];
    double nnz_avg = (double) nnz / (double) n;
    
    double sum = 0;

    for(int i=1; i<partitions;i++){
        sum += rpp*pow((double) ((csrRowPtrL[i*rpp] - csrRowPtrL[(i-1)*rpp])/rpp -nnz_avg),2);
    }
    sum += (n-partitions+1)*pow((double) ((csrRowPtrL[n] - csrRowPtrL[(partitions-1)*rpp])/(n-partitions+1) -nnz_avg),2);
    return sqrt(sum/ (double) (n-1));

}
double get_nnz_stdev( int * csrRowPtrL,  int * csrColIdxL, int n ){


    int nnz = csrRowPtrL[n] - csrRowPtrL[0];
    double nnz_avg = (double) nnz / (double) n;

    double sum = 0;

    for (int i = 0; i < n; ++i)
        sum += pow((double)(csrRowPtrL[i+1] - csrRowPtrL[i]) - nnz_avg, 2);

    return sqrt( sum / (double) (n-1) ) ;
}

int get_nnz_max( int * csrRowPtrL,  int * csrColIdxL, int n ){

    int nnz_max = csrRowPtrL[1] - csrRowPtrL[0];

    for (int i = 1; i < n; ++i)
        if ( nnz_max < csrRowPtrL[i+1] - csrRowPtrL[i] ) nnz_max = csrRowPtrL[i+1] - csrRowPtrL[i];

    return nnz_max;
}

/*float clockElapsed(cudaEvent_t evt_start, cudaEvent_t evt_stop) {
    cudaEventSynchronize(evt_stop);

    float elapsedTime = 0;
    
    cudaEventElapsedTime(&elapsedTime, evt_start, evt_stop);

    return elapsedTime;
}

*/




void run_test(const char * filename, int m, int n, int nnzA, int * csrRowPtrA,  int * csrColIdxA ){
        printf("********************************************************************************************\n");
        printf("                                Datos de la matriz\n");
        printf("\n");
 
    FILE * ftabla;
    ftabla=fopen("res_nnz_est.txt","a");

        fprintf(ftabla,"%s,%i,%i,%i,",filename, m, n, nnzA);

        printf("Filas: %i, NNZ: %i \n", m, nnzA);
        int nnz_max_real = get_nnz_max( csrRowPtrA, csrColIdxA, m);
        printf("Resultado calculado: %i\n", nnz_max_real);
        int res = est_nnz_max(csrRowPtrA, csrColIdxA, m, min(1024,m));
        printf("Particionando:\n");
        // printf("Numero: %i Porcentaje: %i%% Diferencia: %i\n", res, res*100/nnz_max_real, nnz_max_real-res);
        
        fprintf(ftabla,"%i,%i,%.2f,%i|", min(1024,m), res, (float)res/(float)nnz_max_real, nnz_max_real-res);

        // printf("Filas: %i, NNZ: %i \n", m, nnzA);
        // int nnz_max_real = get_nnz_max( csrRowPtrA, csrColIdxA, m);
        // printf("Resultado calculado: %i\n", nnz_max_real);
        res = est_nnz_max(csrRowPtrA, csrColIdxA, m, min(128,m));
        // printf("Particionando:\n");
        // printf("Numero: %i Porcentaje: %.2f Diferencia: %i\n", res, res/nnz_max_real, nnz_max_real-res);
        
        fprintf(ftabla,"%i,%i,%.2f,%i|", min(128,m), res, (float)res/(float)nnz_max_real, nnz_max_real-res);

        // printf("Filas: %i, NNZ: %i \n", m, nnzA);
        // int nnz_max_real = get_nnz_max( csrRowPtrA, csrColIdxA, m);
        // printf("Resultado calculado: %i\n", nnz_max_real);
        res = est_nnz_max(csrRowPtrA, csrColIdxA, m, ceil((float)m/10.0));
        // printf("Particionando:\n");
        // printf("Numero: %i Porcentaje: %.2f Diferencia: %i\n", res, res/nnz_max_real, nnz_max_real-res);
        
        int auxi = ceil((float)m/10.0);
        float auxf = (float)res/(float)nnz_max_real;

        fprintf(ftabla,"%i,%i,%.2f,%i|", auxi , res, auxf, nnz_max_real-res);


        //0 en matrices de 1M
        printf("Bipartición:\n");
        res = est_nnz_maxB(csrRowPtrA, 0, m, 5);
        // printf("Numero: %i Porcentaje: %.2f Diferencia: %i\n", res, res/nnz_max_real,nnz_max_real-res);
        // printf("\n");
        
        fprintf(ftabla,"%i,%.2f,%i|", res, (float)res/(float)nnz_max_real, nnz_max_real-res);


        //< en matrices de 1M
        printf("Bipartición (1/8 de costo):\n");
        res = est_nnz_maxB(csrRowPtrA, 0, m, 2);
        // printf("Numero: %i Porcentaje: %.2f Diferencia: %i\n", res, res/nnz_max_real,nnz_max_real-res);
        // printf("\n");

        fprintf(ftabla,"%i,%.2f,%i|", res, (float)res/(float)nnz_max_real, nnz_max_real-res);


        //< en matrices de 1M
        printf("Bipartición pura:\n");
        res = est_nnz_maxB(csrRowPtrA, 0, m, 0);
        // printf("Numero: %i Porcentaje: %.2f Diferencia: %i\n", res, res/nnz_max_real,nnz_max_real-res);
        // printf("\n");

        fprintf(ftabla,"%i,%.2f,%i|", res, (float)res/(float)nnz_max_real, nnz_max_real-res);


        printf("Desv:\n");

        double stdev = get_nnz_stdev(csrRowPtrA, csrColIdxA, m);
        double stdev_est = est_nnz_stdev(csrRowPtrA, m, min(1024,m)); 
        fprintf(ftabla,"%i,%.2f,%.2f,%.2f,%.2f|",min(1024,m),stdev, stdev_est,(stdev_est/stdev), fabs(stdev-stdev_est));

        // printf("Desviación calculada: %.2f Desviación estimada: %.2f Porcentaje: %.2f Diferencia %.2f\n",stdev, stdev_est, (int)round(stdev_est/stdev), fabs(stdev-stdev_est));
        
        stdev = get_nnz_stdev(csrRowPtrA, csrColIdxA, m);
        stdev_est = est_nnz_stdev(csrRowPtrA, m, min(128,m)); 
        fprintf(ftabla,"%d,%.2f,%.2f,%.2f,%.2f\n",min(128,m),stdev, stdev_est,(stdev_est/stdev), fabs(stdev-stdev_est));


        // printf("*********************************************************************************************\n");
        // printf("\n");

    fclose(ftabla);

    printf("Bye!\n");
}



int main(int argc, char ** argv)
{
    // report precision of floating-point
    printf("---------------------------------------------------------------------------------------------\n");
    char  *precision;
    if (sizeof(VALUE_TYPE) == 4)
    {
        precision = (char *)"32-bit Single Precision";
    }
    else if (sizeof(VALUE_TYPE) == 8)
    {
        precision = (char *)"64-bit Double Precision";
    }
    else
    {
        printf("Wrong precision. Program exit!\n");
        return 0;
    }

    printf("PRECISION = %s\n", precision);
    printf("Benchmark REPEAT = %i\n", BENCH_REPEAT);
    printf("---------------------------------------------------------------------------------------------\n");

    int m, n, nnzA, new_m=-1, new_n=-1;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;

    //ex: ./spmv webbase-1M.mtx
    int argi = 1;

    char  *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }

    if(argc > argi)
    {
        new_m =  atoi(argv[argi++]);
        new_n =  atoi(argv[argi++]);
    }

    printf("-------------- %s --------------\n", filename);

    int device_id = 0;

    if(argc > argi)
    {
	   device_id=atoi(argv[2]); 
       argi++;
    }

    //CUDA_CHK(cudaSetDevice(device_id));
	
    int wpb=WARP_PER_BLOCK;
    if(argc > argi)
    {
        wpb = atoi(argv[3]);
        argi++; 
    }

    printf("WARPS PER BLOCK = %i.\n",wpb);
    

    // read matrix from mtx file
    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnzA_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if ( mm_is_complex( matcode ) )
    {
        printf("Sorry, data type 'COMPLEX' is not supported.\n");
        return -3;
    }

    char * pch,* pch1;
    pch = strtok (filename,"/");
    while (pch != NULL){
        pch1 = pch;
        pch = strtok (NULL, "/");
    }

    pch = strtok (pch1, ".");
    filename=pch;

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
    if (ret_code != 0)
        return -4;


    // if (n != m)
    // {
    //     printf("Matrix is not square.\n");
    //     return -5;
    // }


    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric = 1;
        printf("input matrix is symmetric = true\n");
    }
    else
    {
        printf("input matrix is symmetric = false\n");
    }

    int *csrRowPtrA_counter = (int *)malloc((m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));

    int *csrRowIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    int *csrColIdxA_tmp = (int *)malloc(nnzA_mtx_report * sizeof(int));
    VALUE_TYPE *csrValA_tmp    = (VALUE_TYPE *)malloc(nnzA_mtx_report * sizeof(VALUE_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnzA_mtx_report; i++)
    {
        int idxi, idxj;
        double fval;
        int ival;
        int returnvalue;

        if (isReal)
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtrA_counter[idxi]++;
        csrRowIdxA_tmp[i] = idxi;
        csrColIdxA_tmp[i] = idxj;
        csrValA_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtrA_counter
    int old_val, new_val;

    old_val = csrRowPtrA_counter[0];
    csrRowPtrA_counter[0] = 0;
    for (int i = 1; i <= m; i++)
    {
        new_val = csrRowPtrA_counter[i];
        csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i-1];
        old_val = new_val;
    }

    nnzA = csrRowPtrA_counter[m];
    csrRowPtrA = (int *)malloc((m+1) * sizeof(int));
    memcpy(csrRowPtrA, csrRowPtrA_counter, (m+1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m+1) * sizeof(int));

    csrColIdxA = (int *)malloc(nnzA * sizeof(int));
    csrValA    = (VALUE_TYPE *)malloc(nnzA * sizeof(VALUE_TYPE));

    if (isSymmetric)
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

                offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
                csrColIdxA[offset] = csrRowIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
            }
        }
    }
    else
    {
        for (int i = 0; i < nnzA_mtx_report; i++)
        {
            int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
            csrColIdxA[offset] = csrColIdxA_tmp[i];
            csrValA[offset] = csrValA_tmp[i];
            csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
        }
    }

    printf("main::1\n");fflush(0);

/*
    // free tmp space
    free(csrColIdxA_tmp);
    free(csrValA_tmp);
    free(csrRowIdxA_tmp);
    free(csrRowPtrA_counter);

    // a small matrix
    free(csrColIdxA);
    free(csrValA);
    free(csrRowPtrA);

    m = n = 8;
    nnzA = 17;
    csrRowPtrA = (int *)malloc(sizeof(int) * (m+1));
    csrColIdxA = (int *)malloc(sizeof(int) * nnzA);
    csrValA    = (VALUE_TYPE *)malloc(nnzA * sizeof(VALUE_TYPE));
    csrRowPtrA[0] = 0; csrRowPtrA[1] = 1; csrRowPtrA[2] = 2; csrRowPtrA[3] = 4; csrRowPtrA[4] = 6; 
    csrRowPtrA[5] = 10; csrRowPtrA[6] = 12; csrRowPtrA[7] = 15; csrRowPtrA[8] = nnzA;
    

    csrColIdxA[0] = 0;  csrColIdxA[1] = 1;  csrColIdxA[2] = 1;  csrColIdxA[3] = 2;  csrColIdxA[4] = 0;  
    csrColIdxA[5] = 3;  csrColIdxA[6] = 1;  csrColIdxA[7] = 2;  csrColIdxA[8] = 3;  csrColIdxA[9] = 4;  
    csrColIdxA[10] = 3;  csrColIdxA[11] = 5;  csrColIdxA[12] = 2;  csrColIdxA[13] = 5;  csrColIdxA[14] = 6;  
    csrColIdxA[15] = 6;  csrColIdxA[16] = 7;  
    // a small matrix stop
*/
    //printf("input matrix A: ( %i, %i ) nnz = %i\n", m, n, nnzA);

    // extract L with the unit-lower triangular sparsity structure of A
    int nnzL = 0;
    int *csrRowPtrL_tmp = (int *)malloc((m+1) * sizeof(int));
    int *csrColIdxL_tmp = (int *)malloc(nnzA * sizeof(int));
    VALUE_TYPE *csrValL_tmp    = (VALUE_TYPE *)malloc(nnzA * sizeof(VALUE_TYPE));

    int nnz_pointer = 0;
    csrRowPtrL_tmp[0] = 0;
    for (int i = 0; i < m; i++)
    {
        for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
        {
            if (csrColIdxA[j] < i)
            {
                csrColIdxL_tmp[nnz_pointer] = csrColIdxA[j];
                csrValL_tmp[nnz_pointer] = 1.0; //csrValA[j];
                nnz_pointer++;
            }
            else
            {
                break;
            }
        }

        csrColIdxL_tmp[nnz_pointer] = i;
        csrValL_tmp[nnz_pointer] = 1.0;
        nnz_pointer++;

        csrRowPtrL_tmp[i+1] = nnz_pointer;
    }

    nnzL = csrRowPtrL_tmp[m];
    //printf("A's unit-lower triangular L: ( %i, %i ) nnz = %i\n", m, n, nnzL);

    csrColIdxL_tmp = (int *)realloc(csrColIdxL_tmp, sizeof(int) * nnzL);
    csrValL_tmp = (VALUE_TYPE *)realloc(csrValL_tmp, sizeof(VALUE_TYPE) * nnzL);

    // run serial syncfree SpTS as a reference
    printf("---------------------------------------------------------------------------------------------\n");
    
    run_test(filename, m, n, nnzL, csrRowPtrL_tmp, csrColIdxL_tmp);

//    spts_syncfree_serialref(csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp, m, n, nnzL);

    // set device
    //int device_id = 0;
  /*  cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    printf("---------------------------------------------------------------------------------------------\n");
    printf("Device [ %i ] %s @ %4.2f MHz\n", device_id, deviceProp.name, deviceProp.clockRate * 1e-3f);
*/
    // run cuda syncfree SpTRSV
    // printf("---------------------------------------------------------------------------------------------\n");
    // spts_syncfree_cuda(csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp, m, n, nnzL, filename, wpb);
    // printf("---------------------------------------------------------------------------------------------\n");

    //test_solve_L_analysis_multirow( pch, csrRowPtrL_tmp, csrColIdxL_tmp, csrValL_tmp, n);
    int cantidadDeVeces;
    /*
    while(1){
        printf("Resultado calculado:\n");
        printf("%i\n", get_nnz_max( csrRowPtrL_tmp, csrColIdxL_tmp, m));
        printf("¿Cantidad de veces?\n");
        scanf("%d", &cantidadDeVeces);
        if(cantidadDeVeces==0)
            break;
        int tony = est_nnz_max(csrRowPtrL_tmp, csrColIdxL_tmp, m, cantidadDeVeces);
        printf("Particionando:\n");
        printf("%i\n", tony);
        printf("¿N?\n");
        int number123;
        scanf("%i", &number123);
        printf("Bipartición:\n");
        tony = est_nnz_maxB(csrRowPtrL_tmp, 0, m+1, number123);
        printf("%i\n", tony);
        printf("\n");
        printf("---------------------------------------------------------------------------------------------\n");
        printf("\n");
        }*/

    	//prueba(csrRowPtrA,csrValA);



    // if ( new_m != -1 && new_n != -1 ){
    //     if (new_m == new_n){
    //         run_test(filename, m, n, nnzA, csrRowPtrA, csrColIdxA);
    //     }else{
    //         m = min(new_m, new_n);

    //         int nnzNew = csrRowPtrA[m] - csrRowPtrA[0];

    //         int * csrRowPtrA_new = (int *) malloc((m+1) * sizeof(int));
    //         int * csrColIdxA_new = (int *) malloc((nnzNew) * sizeof(int));
    //         VALUE_TYPE * csrValA_new = (VALUE_TYPE *) malloc((nnzNew) * sizeof(VALUE_TYPE));

    //         printf("main::2\n");fflush(0);

    //         memcpy(csrRowPtrA_new, csrRowPtrA, (m+1) * sizeof(int));
    //         memcpy(csrColIdxA_new, csrColIdxA, (nnzNew) * sizeof(int));
    //         memcpy(csrValA_new, csrValA, (nnzNew) * sizeof(VALUE_TYPE));

    //         printf("main::3\n");fflush(0);


    //         free(csrRowPtrA);
    //         free(csrColIdxA);
    //         free(csrValA);

    //         printf("main::4\n");fflush(0);

    //         csrRowPtrA = csrRowPtrA_new;
    //         csrColIdxA = csrColIdxA_new;
    //         csrValA    = csrValA_new;

    //         nnzA = nnzNew;
        

    //         if (new_m < new_n){ 
    //             printf("run test: %d x %d\n", m, n ); fflush(0);
                
    //             run_test(filename, m, n, nnzA, csrRowPtrA, csrColIdxA);
    //         } else {
    //             n = new_m;
    //             m = new_n;

    //             int * csrRowPtrA_tr = (int *) malloc((n+1) * sizeof(int));
    //             int * csrColIdxA_tr = (int *) malloc((nnzA) * sizeof(int));

    //             printf("main::5\n");fflush(0);

    //             matrix_transposition( m, n, nnzA,
    //                                  csrRowPtrA, csrColIdxA, csrValA,
    //                                  csrColIdxA_tr, csrRowPtrA_tr, csrValA);

    //             printf("run test: %d x %d\n", n, m );

    //             run_test(filename, n, m, nnzA, csrRowPtrA_tr, csrColIdxA_tr);

    //             free(csrRowPtrA_tr);
    //             free(csrColIdxA_tr);
    //         }
    //     }
    // }else{
    //     run_test(filename, m, n, nnzA, csrRowPtrA, csrColIdxA);    
    // }


    // // done!
    // free(csrColIdxA);
    // free(csrValA);
    // free(csrRowPtrA);

    free(csrColIdxL_tmp);
    free(csrValL_tmp);
    free(csrRowPtrL_tmp);

    return 0;
}
