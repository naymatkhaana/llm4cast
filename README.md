**Pretraining**
python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/illness/ --data_path national_illness_24_ilitotal.csv --model_id tmmodel2_p --model GPT4TS_tinyllama_qlora --data customoriginfv2 --features MS --seq_len 104 --label_len 0 --pred_len 12 --e_layers 7 --d_layers 1 --factor 3 --enc_in 7 --dec_in 7 --c_out 7 --des 'Exp' --n_heads 8 --batch_size 128 --patch_len 12 --stride 2 --num_workers 20 --gpt_layers 30 --learning_rate 0.000001


**Finetuning**
python -u run.py   --task_name long_term_forecast   --is_training 0   --root_path ./dataset/illness/   --data_path national_illness_24.csv   --model_id tmmodel2_p   --model GPT4TS_tinyllama_qlora   --data customorig   --features MS   --seq_len 104   --label_len 0  --pred_len 12 --e_layers 7  --d_layers 1   --factor 3   --enc_in 7   --dec_in 7   --c_out 7   --des 'Exp'   --n_heads 8 --batch_size 32 --patch_len 12 --stride 2 --num_workers 20 --gpt_layers 30 --learning_rate 0.000001 --target ILITOTAL --full_shot 1 --full_shot_path ./checkpoints/long_term_forecast_tmmodel2_p_GPT4TS_tinyllama_qlora_customoriginfv2_ftMS_sl104_ll0_pl12_dm2048_nh8_el7_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0
