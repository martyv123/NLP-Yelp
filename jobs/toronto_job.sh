#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vo.ma@northeastern.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --job-name=toronto_job
#SBATCH --mem=12G
#SBATCH --partition=short
#SBATCH --output=toronto_job.log

echo "Loading the Conda module"
module load anaconda3/3.7

echo "Activate Conda environment"
source activate yelp

echo "Start the Python script"
python -W ignore cosine_similarity.py toronto/toronto_businesses.csv toronto/toronto_reviews.csv toronto/toronto_users.csv toronto/toronto_similarities.csv toronto/finished_toronto_businesses.csv
