#!/bin/bash
#SBATCH --job-name=mitrabajo
#SBATCH --ntasks=1
#SBATCH --mem=512
#SBATCH --time=00:01:00
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=mi@correo
#SBATCH -o salida.out

export MKL_VERBOSE=1
export MKL_NUM_THREADS=4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/lib/intel64_lin:/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin

source /etc/profile.d/modules.sh

mat_dir=/clusteruy/home/edufrechou/matrices 
# matrices=$(find $mat_dir -maxdepth 1 |  grep -E ".mtx")
# matrices=$(cat matrices2.txt)

cd ~/sptrsv

make
$1 $2
