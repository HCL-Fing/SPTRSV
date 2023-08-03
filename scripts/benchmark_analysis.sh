#!/bin/bash


# values for the drop tolerance
# and condest  
factor_tol="0.01"
condest=5 #100

mat_dir=/home/gpgpu/users/edufrechou/matrices/sptrsv 

# select matrices
matrices=$(find $mat_dir -maxdepth 1 |  grep -E ".mtx")
#cd $mat_dir
#matrices= eval 'cat ../bajar_matrices.sh'      #$(find $mat_dir |  grep -E ".tar.gz")
#cd $ilupack_dir
#echo $matrices

#
#echo $matrices
echo "Comienzo" >> resultados.txt
echo "single" >> resultados.txt
for device in 0 
do

	for wpb in 4 8 16 32 
	do

		for z in $matrices
		do
		timeout 5m ./sptrsv_analysis_float $z $device $wpb 
		done
	done

	echo "double" >> resultados.txt


	for wpb in  4 8 16 32
	do

		for z in $matrices
		do
		timeout 5m ./sptrsv_analysis_double $z $device $wpb 
		done
	done

done

echo "Fin" >> resultados.txt
