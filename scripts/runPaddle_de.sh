
# python -u ../train.py --exp_name unsupMTDiscCLF_ende --dump_path ./dumped/ --reload_model './pretrain/pretrain_ende/mlm_ende_mt_1024_20190926.pth,./pretrain/pretrain_ende/mlm_ende_mt_1024_20190926.pth' --data_path ./data/processed/de-en/ --train_dis True --train_dae True --train_bt True --lgs 'de-en' --ae_steps 'de,en' --bt_steps 'de-en-de,en-de-en' --dis_steps 1 --max_len 100 --word_shuffle 3 --word_dropout 0.1 --lambda_dis 4 --lambda_clf 16 --data_category books --lambda_div 0.5 --clf_mtv True --clf_steps 10 --clf_batch_norm True --clf_attention False --clf_dropout 0.1 --clf_layers 2 --word_blank 0.1 --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --optimizer adam,lr=0.00002 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 600 --batch_size 32 --bptt 256 --max_epoch 100000 --epoch_size 64000 --stopping_criterion 'valid_en_clf_acc,20' --validation_metrics 'valid_en_clf_acc'

# Usage: bash runPaddle_de.sh --clf_steps 50 --num_gpu 1 --exp_name unsupMTDiscCLF_ende --data_category books (dvd or music) --train_dis True --train_encdec True --train_bt True --clf_atten False --clf_mtv True --tokens_per_batch 600

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  --gpuid)
    GID="$2"
    shift 2
    ;;
  --exp_name)
    EXPNAME="$2"
    shift 2
    ;;
  --data_category)
    DATA_CAT="$2"
    shift 2
    ;;
  --train_dis)
    TRDIS="$2"
    shift 2
    ;;
  --train_encdec)
    TRENCDEC="$2"
    shift 2
    ;;
  --train_bt)
    TRBT="$2"
    shift 2
    ;;
  --clf_atten)
    CLFATTN="$2"
    shift 2
    ;;
  --clf_mtv)
    CLFMTV="$2"
    shift 2
    ;;
  --tokens_per_batch)
    TPB="$2"
    shift 2
    ;;
  --clf_steps)
    CLFSTEPS="$2"; shift 2
    ;;
  *)
    POSITIONAL+=("$1")
    shift
    ;;
  esac
done
set -- "${POSITIONAL[@]}"
# export CUDA_VISIBLE_DEVICES=$GID
export CUDA_LAUNCH_BLOCKING=1

python -u -W ignore train.py \
  --exp_name $EXPNAME \
  --dump_path ./dumped/ \
  --data_path ./data/processed/de-en/ \
  --reload_model './pretrain/pretrain_deen/mlm_ende_mt_1024_20190926.pth,./pretrain/pretrain_deen/mlm_ende_mt_1024_20190926.pth' \
  --train_dis $TRDIS \
  --train_dae $TRENCDEC \
  --train_bt $TRBT \
  --lgs 'de-en' \
  --ae_steps 'de,en' \
  --bt_steps 'de-en-de,en-de-en' \
  --dis_steps 1 \
  --max_len 100 \
  --word_shuffle 3 \
  --word_dropout 0.1 \
  --lambda_dis 8 \
  --lambda_clf 4 \
  --data_category $DATA_CAT \
  --clf_output_dim 2 \
  --lambda_div 0.5 \
  --clf_mtv $CLFMTV \
  --clf_steps $CLFSTEPS \
  --clf_batch_norm True \
  --clf_attention $CLFATTN \
  --clf_dropout 0.1 \
  --clf_layers 2 \
  --dec_layers 6 \
  --word_blank 0.1 \
  --lambda_ae '0:1,100000:0.1,300000:0' \
  --encoder_only false \
  --emb_dim 1024 \
  --n_layers 6 \
  --n_heads 8 \
  --optimizer adam,lr=0.00001 \
  --dropout 0.1 \
  --attention_dropout 0.1 \
  --gelu_activation true \
  --tokens_per_batch $TPB \
  --batch_size 32 \
  --bptt 256 \
  --max_epoch 100000 \
  --epoch_size 100000 \
  --stopping_criterion 'valid_en_clf_acc,5' \
  --validation_metrics 'valid_en_clf_acc'