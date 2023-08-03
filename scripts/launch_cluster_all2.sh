#!/bin/bash
#SBATCH --job-name=nnz_est_all
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=normal
# SBATCH --qos=gpu
# SBATCH --gres=gpu:1
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=mi@correo
#SBATCH -o salida.out


export MKL_VERBOSE=1
export MKL_NUM_THREADS=4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/lib/intel64_lin

source /etc/profile.d/modules.sh

cd ~/sptrsv

# mat_dir=/clusteruy/home/edufrechou/matrices 
# matrices=$(find $mat_dir -maxdepth 1 |  grep -E ".mtx")
# matrices=$(cat scripts/lista_goodwin.csv)
# matrices=$(cat scripts/lista_aleat.csv)
matrices=$(cat scripts/lista.txt)



#/clusteruy/home/edufrechou/gpgpu/samples/1_Utilities/deviceQuery/deviceQuery

#for m in $matrices; do
#	echo ./sptrsv_float $m
#done

# for m in $matrices; do
# 	./sptrsv_double /clusteruy/home/edufrechou/matrices/$m.mtx
# done

matdir=/clusteruy/home/edufrechou/matrices

for m in $matrices
do
#  	grupo_mat=$(echo $m | sed 's/\(.*\);\(.*\);\(.*\);\(.*\)/\1/');
#  	nombre_mat=$(echo $m | sed 's/\(.*\);\(.*\);\(.*\);\(.*\)/\2/');
#  	eme=$(echo $m | sed 's/\(.*\);\(.*\);\(.*\);\(.*\)/\3/');
#  	ene=$(echo $m | sed 's/\(.*\);\(.*\);\(.*\);\(.*\)/\4/');

#  	cd $matdir

# #	if [ ! -f $m.tar.gz ]; then
# 	stat $nombre_mat.mtx || wget https://sparse.tamu.edu/MM/$grupo_mat/$nombre_mat.tar.gz
# 	stat $nombre_mat.tar.gz && tar xvzf $nombre_mat.tar.gz && mv $nombre_mat/$nombre_mat.mtx . && rm -rf $nombre_mat.tar.gz
# #	fi

# 	cd ~/sptrsv


	#timeout 2m /home/gpgpu/users/edufrechou/sptrsv/sptrsv_double $nombre_mat/$nombre_mat.mtx
	timeout 2m ./sptrsv_double $matdir/$m.mtx || echo "$m" >> fallos.txt

	# rm -f resultados_tirar.txt
	# rm -rf $nombre_mat.tar.gz
	# rm -rf $nombre_mat
done
