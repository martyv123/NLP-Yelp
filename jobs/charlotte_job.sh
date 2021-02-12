#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vo.ma@northeastern.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --job-name=charlotte_job
#SBATCH --mem=12G
#SBATCH --partition=short
#SBATCH --output=charlotte_job.log

echo "Loading the Conda module"
module load anaconda3/3.7

echo "Activate Conda environment"
source activate yelp

echo "Start the Python script"
python -W ignore cosine_similarity.py charlotte/charlotte_businesses.csv charlotte/charlotte_reviews.csv charlotte/charlotte_users.csv charlotte/charlotte_similarities.csv charlotte/finished_charlotte_businesses.csv
