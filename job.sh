#!/bin/bash
#
# to run:
# chmod +x job.sh
# ./job.sh
#
#
# SLURM directives
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --time=11:59:00
#SBATCH --job-name=myjob
#SBATCH --output=./log/output_%j.txt
#SBATCH --error=./log/error_%j.txt
#
module load anaconda3/2022.05
module load python/3.8.1
pip install -r requirements.txt

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <parameter1> <parameter2>"
  exit 1
fi
game=$1
crit=$2
command="python active_train.py --game $game --criteria $crit --n_ini 10 --n_instances 1"
echo $command
$command > logs/$crit_$game.txt
#
#
# games=("cave" "mario" "supercat" "tomb")
# criteria=("random" "entropy" "margin" "uncertainty")
# for game in "${games[@]}"; do
#     for crit in "${criteria[@]}"; do
#         echo "Game: $game, Criteria: $crit"
#         command="python active_train.py --game $game --criteria $crit --n_ini 10 --n_instances 1"
#         echo $command
#         $command
#     done
# done




