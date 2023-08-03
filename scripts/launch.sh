#!/bin/bash
#SBATCH --job-name=nnz_est_all
#SBATCH --ntasks=1
#SBATCH --mem=512
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

# mat_dir=/clusteruy/home/edufrechou/matrices 
# matrices=$(find $mat_dir -maxdepth 1 |  grep -E ".mtx")
matrices=$(cat matrices.txt)

cd ~/sptrsv

#/clusteruy/home/edufrechou/gpgpu/samples/1_Utilities/deviceQuery/deviceQuery

#for m in $matrices; do
#	echo ./sptrsv_float $m
#done

for m in $matrices; do
	./sptrsv_double /clusteruy/home/edufrechou/matrices/$m.mtx
done
