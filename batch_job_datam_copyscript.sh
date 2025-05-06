#!/bin/bash
#SBATCH --job-name=jobts              # Job name
#SBATCH --partition=batch            # Partition (queue) name
#SBATCH --ntasks=4                    # Run a single task
#SBATCH --cpus-per-task=2             # Number of CPU cores per task
#SBATCH --mem=50gb                    # Job memory request
#SBATCH --time=150:00:00               # Time limit hrs:min:sec
#SBATCH --output=amber.%j.out         # Standard output log
#SBATCH --error=amber.%j.err          # Standard error log

#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=fs47816@uga.edu  # Where to send mail (change username@uga.edu to your email address)

cp -r /scratch/fs47816/workdir/sample_scripts/time_series_dl/time-series-v2 /scratch/fs47816/workdir/sample_scripts/time_series_dl/time-series-v4

 
