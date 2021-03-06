#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vo.ma@northeastern.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --job-name=phoenix_job
#SBATCH --mem=12G
#SBATCH --partition=short
#SBATCH --output=phoenix_job.log

echo "Loading the Conda module"
module load anaconda3/3.7

echo "Activate Conda environment"
source activate yelp

echo "Start the Python script"
python -W ignore cosine_similarity.py phoenix/phoenix_businesses.csv phoenix/phoenix_reviews.csv phoenix/phoenix_users.csv phoenix/phoenix_similarities.csv phoenix/finished_phoenix_businesses.csv
