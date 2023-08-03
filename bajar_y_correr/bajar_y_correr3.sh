#/bin/sh!

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gpgpu/software/magma/magma-2.5.2/lib

matrices=$(cat lista40.txt)

for m in $matrices
do
 	grupo_mat=$(echo $m | sed 's/\(.*\);\(.*\);\(.*\);\(.*\)/\1/');
 	nombre_mat=$(echo $m | sed 's/\(.*\);\(.*\);\(.*\);\(.*\)/\2/');
 	eme=$(echo $m | sed 's/\(.*\);\(.*\);\(.*\);\(.*\)/\3/');
 	ene=$(echo $m | sed 's/\(.*\);\(.*\);\(.*\);\(.*\)/\4/');

 	
	if [ ! -f $m.tar.gz ]; then
		wget https://sparse.tamu.edu/MM/$grupo_mat/$nombre_mat.tar.gz
	fi

	tar xvzf $nombre_mat.tar.gz

	 timeout 2m /home/manuel/Downloads/sptrsv-max_nnz_est/sptrsv_double $nombre_mat/$nombre_mat.mtx

#	rm -f resultados_tirar.txt
	rm -rf $nombre_mat.tar.gz
	rm -rf $nombre_mat
done
