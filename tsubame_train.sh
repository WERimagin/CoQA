#!/bin/sh
##n current working directory
#$ -cwd
## Resource type F: qty 2
#$ -l f_node=2
## maximum run time
#$ -l h_rt=1:30:00
## output filename
#$ -N sample

. /etc/profile.d/modules.sh
module load python/3.6.5
module load intel
module load cuda
module load openmpi
#実行
source imagin/bin/activate
python3 train.py
