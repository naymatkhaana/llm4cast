#!/bin/bash
#SBATCH --job-name=jobts              # Job name
#SBATCH --partition=gpu_p            # Partition (queue) name
#SBATCH --gres=gpu:A100:1                  # Requests one GPU device
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=20             # Number of CPU cores per task
#SBATCH --mem=200gb                    # Job memory request
#SBATCH --time=150:00:00               # Time limit hrs:min:sec
#SBATCH --output=amber.%j.out         # Standard output log
#SBATCH --error=amber.%j.err          # Standard error log

#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=fs47816@uga.edu  # Where to send mail (change username@uga.edu to your email address)

cd /scratch/fs47816/workdir/sample_scripts/time_series_dl/time-series-v2/Time-Series-Library


ml Python/3.9.5-GCCcore-10.3.0


export CUDA_VISIBLE_DEVICES=0

python -u run.py   --task_name long_term_forecast   --is_training 1   --root_path ./dataset/illness/   --data_path national_illness.csv   --model_id m_model_v9_v2_v_m   --model GPT4TS_tinyllama_qdora   --data datam_short   --features MS   --seq_len 36   --label_len 4  --pred_len 60 --e_layers 7  --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --n_heads 8 --batch_size 2048 --patch_len 12  --stride 2  --gpt_layers 30 --num_workers 20 --learning_rate 0.00001 --full_shot 1 --full_shot_path ./checkpoints/long_term_forecast_tmmodel2_GPT4TS_tinyllama_qdora_customoriginfv2_ftMS_sl36_ll0_pl60_dm2048_nh8_el7_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0 
