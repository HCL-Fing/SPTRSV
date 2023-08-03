#!/bin/bash


mat_dir=/home/gpgpu/users/edufrechou/matrices/sptrsv/nuevas 

# select matrices
#matrices=$(find $mat_dir -maxdepth 1 |  grep -E ".mtx")
matrices=$(cat matrices.txt)
#cd $mat_dir
#matrices= eval 'cat ../bajar_matrices.sh'      #$(find $mat_dir |  grep -E ".tar.gz")
#cd $ilupack_dir
#echo $matrices

#
#echo $matrices
echo "Comienzo" >> resultados.txt
echo "doble" >> resultados.txt
#for device in 0 
#do

#	for wpb in 28 #4 8 16 32 
#	do
		for z in $matrices
		do
		timeout 5m ./sptrsv_double $mat_dir/$z.mtx
		done
#	done

#	echo "double" >> resultados.txt


#	for wpb in  4 8 16 32
#	do

#		for z in $matrices
#		do
#		timeout 1m ./sptrsv_float $z
#		done
#	done

#done

echo "Fin" >> resultados.txt
