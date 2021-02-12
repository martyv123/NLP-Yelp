#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vo.ma@northeastern.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --job-name=boston_op_job
#SBATCH --mem=12G
#SBATCH --partition=short
#SBATCH --output=boston_op_job.log

echo "Loading the Conda module"
module load anaconda3/3.7

echo "Activate Conda environment"
source activate yelp

echo "Start the Python script"
python -W ignore open_table.py open_table/boston/boston_businesses.csv open_table/boston/boston.csv open_table/boston/boston.csv open_table/boston/boston_op_similarities.csv open_table/boston/finished_boston_businesses.csv
