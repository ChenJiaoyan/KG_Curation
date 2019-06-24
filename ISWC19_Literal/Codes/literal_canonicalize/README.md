Requirements:
    Tensorflow 1.13, sparql-client 3.4, gensim 3.7.1, Pattern 3.6

----------------------

To see the final results directly, please run:
    python -u typing.py --data_name RData --data_file Data/RData_Clean.csv --out_score_file RData_Scores_AttBiRNN_FTF.json --type_gt Data/RData_Type.json
    OR
    python -u typing.py --data_name SData --data_file Data/SData_Clean.csv --out_score_file SData_Scores_AttBiRNN_FTF.json --type_gt Data/SData_Type_fixed.json

You can change  --typing_method to H or I, set --kappa to 0 or -0.1
The default the results are for S-Lite, scores are predicted by fine tuned AttBiRNN

----------------------

To reproduce all the steps:

   just set word2vec model (trained by gensim, see Word2Vec/) in quick_run.sh and run it

   Or run step by step with the following instructions:

    -- Step #1: pretrain.py (pretrain classifiers, set --sequence_len to '12,4,12' for SData and '12,4,15' for RData, set --nn_dir and --data_file and --nn_type)

    -- Step #2: classes.py (generate candidate classes, set --entity_mask_type to 'YES' for SData and 'NO' for RData, set --data_name, --data_file and --out_file)

    -- Step #3: samples.py (generate particular samples, set --entity_mask_type to 'YES' for SData and 'NO' for RData, set --data_name, --data_file and --particular_sample_file, set --use_fixed_entity_class to 'YES' for refined samples and 'NO' otherwise)

    -- Step #4: prediction.py (fine tune if --need_finetune is set to Yes, and predict scores, set --data_name, --data_file and --particular_sample_file and --out_score_file, set --sequence_len to '12,4,12' for SData and '12,4,15' for RData)

    -- Step #5: typing.py (type decision making, set --data_name, --data_file, --out_score_file and --type_gt, set --typing_method, --iota_range and --kappa)

    -- Step #6: enntity_lookup.py (entity lookup, only works for RData, the results need to be checked manually, set --data_file, --score_file, --cache_entity_file, --cache_class_file, --out_entity_file, --threshold and --filter_by_types)


Other parameters to set:

    --wv_model_dir: directory of word2vec model trained by gensim

    --for those parameters not specifically pointed out, you can use the default
