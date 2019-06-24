#!/usr/bin/env bash

python -u pretrain.py --wv_model_dir ~/word2vec/w2v_model/enwiki_model --nn_dir AttBiRNN-50-RData --entity_mask_type NO --sequence_lens 12,4,15 --data_file Data/RData_Clean.csv

# comment it if the class file (S)RData_Classes.json exists
python -u classes.py --data_name RData --data_file Data/RData_Clean.csv --out_file RData_Classes.json --entity_mask_type NO

# comment it if tthe particular sample files (S)RData_PSamples.json and (S)RData_PSamples_fixed.json exist
python -u samples.py --data_name RData --data_file Data/RData_Clean.csv --entity_mask_type NO --use_fixed_entity_class YES --particular_sample_file RData_PSamples_fixed.json 

python -u prediction.py --wv_model_dir ~/word2vec/w2v_model/enwiki_model --data_file Data/RData_Clean.csv --out_score_file RData_Scores_AttBiRNN_FTF.json --need_finetune Yes --particular_sample_file RData_PSamples_fixed.json --sequence_lens 12,4,15 --nn_dir AttBiRNN-50-RData

python -u typing.py --typing_method I --data_name RData --data_file Data/RData_Clean.csv --out_score_file RData_Scores_AttBiRNN_FTF.json --type_gt Data/RData_Type.json
python -u typing.py --typing_method H --kappa 0 --data_name RData --data_file Data/RData_Clean.csv --out_score_file RData_Scores_AttBiRNN_FTF.json --type_gt Data/RData_Type.json
python -u typing.py --typing_method H --kappa -0.1 --data_name RData --data_file Data/RData_Clean.csv --out_score_file RData_Scores_AttBiRNN_FTF.json --type_gt Data/RData_Type.json
