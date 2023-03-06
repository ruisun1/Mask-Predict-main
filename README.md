# Mask-Predict




### Preprocess

text=PATH_YOUR_DATA

output_dir=PATH_YOUR_OUTPUT
'''
python preprocess.py --source-lang wro --target-lang cor --trainpref $text/train --validpref $text/valid --testpref $text/test  --destdir ${output_dir}/data-bin  --workers 60  --srcdict ${model_path}/maskPredict_${src}_${tgt}/dict.${src}.txt --tgtdict ${model_path}/maskPredict_${src}_${tgt}/dict.${tgt}.txt
'''
### Train


model_dir=PLACE_TO_SAVE_YOUR_MODEL

python train.py ${output_dir}/data-bin --arch bert_transformer_seq2seq_1e --share-all-embeddings --criterion confusion_loss3 --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates 10000 --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --task translation_spellcheck_final --max-tokens 8192 --weight-decay 0.01 --dropout 0.3 --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512  --fp16 --max-source-positions 10000 --max-target-positions 10000 --max-update 300000 --seed 0 --save-dir ${model_dir}

### Evaluation


python  generate_cmlm_sc_1encoder.py  ${output_dir}/data-bin  --path ${model_dir}/checkpoint_best_average.pt  --task translation_spellcheck_final --remove-bpe --max-sentences 20 --decoding-iterations 10  --decoding-strategy mask_predict_sc_1e



