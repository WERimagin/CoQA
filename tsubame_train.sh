#!/bin/sh
##n current working directory
#$ -cwd
## Resource type F: qty 2
#$ -l f_node=1
## maximum run time
#$ -l h_rt=0:10:00
## output filename
#$ -N sample

. /etc/profile.d/modules.sh
module load python/3.6.5
module load intel
module load cuda
module load openmpi
#実行
source ../imagin/bin/activate
export PYTHONPATH=/home/6/15B06641/CoQA
python3 train.py
