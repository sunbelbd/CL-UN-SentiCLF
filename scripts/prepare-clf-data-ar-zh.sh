# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# usage: nohup bash prepare-clf-data-ar-zh.sh --src en --tgt zh --reload_codes ./data/processed/en-zh/codes --reload_vocab ./data/processed/en-zh/vocab.en-zh --product hotel >logs/process_enzh_hotel_20191202.log 2>&1 &

# usage: nohup bash prepare-clf-data-ar-zh.sh --src en --tgt ar --reload_codes ./data/processed/ar-en/codes --reload_vocab ./data/processed/ar-en/vocab.ar-en --product BBN >logs/process_aren_BBN_20191202.log 2>&1 &

# similar process with other langs, but have to process monolingual train.en as well.

# vocab.en-zh has 236415 subwords, select top 60k

# use 650k yelp training text as monolingual EN source -> train.en, which has been processed
# use 160k Chinese hotel training text as monolingual ZH source -> train.zh
# Use 6k validation set of Arab sentiment data as monolingual AR source -> train.ar

########## finetune xlm 100
# usage: nohup bash prepare-clf-data-ar-zh.sh --src en --tgt zh --reload_codes ./pretrain/pretrain_xlm_100/codes_xnli_100 --reload_vocab ./pretrain/pretrain_xlm_100/vocab_xnli_100 --product hotel >logs/process_enzh_hotel_20191202_xlm100.log 2>&1 &

# usage: nohup bash prepare-clf-data-ar-zh.sh --src en --tgt ar --reload_codes ./pretrain/pretrain_xlm_100/codes_xnli_100 --reload_vocab ./pretrain/pretrain_xlm_100/vocab_xnli_100 --product BBN >logs/process_aren_BBN_20191202_xlm100.log 2>&1 &

########## finetune xlm 17
# usage: nohup bash prepare-clf-data-ar-zh.sh --src en --tgt zh --reload_codes ./pretrain/pretrain_xlm_17/codes_xnli_17 --reload_vocab ./pretrain/pretrain_xlm_17/vocab_xnli_17 --product hotel >logs/process_enzh_hotel_20191202_xlm17.log 2>&1 &

# usage: nohup bash prepare-clf-data-ar-zh.sh --src en --tgt ar --reload_codes ./pretrain/pretrain_xlm_17/codes_xnli_17 --reload_vocab ./pretrain/pretrain_xlm_17/vocab_xnli_17 --product BBN >logs/process_aren_BBN_20191202_xlm17.log 2>&1 &

######### Regenerate FR, DE monolingual data using unlabeled data, not the 5M unmt training data

# nohup bash prepare-clf-data-ar-zh.sh --src en --tgt fr --reload_codes ./data/processed/en-fr/codes --reload_vocab ./data/processed/en-fr/vocab.en-fr --product books >logs/process_enfr_mono_20191206.log 2>&1 &

# nohup bash prepare-clf-data-ar-zh.sh --src en --tgt de --reload_codes ./data/processed/de-en/codes --reload_vocab ./data/processed/de-en/vocab.de-en --product dvd >logs/process_deen_mono_20191206.log 2>&1 &

set -e

#
# Data preprocessing configuration
#
N_MONO=5000000 # number of monolingual sentences for each language
CODES=60000    # number of BPE codes
N_THREADS=16   # number of threads in data preprocessing

#
# Read arguments
# --product is specially designed for amazon data.
#
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  --src)
    SRC="$2"
    shift 2
    ;;
  --tgt)
    TGT="$2"
    shift 2
    ;;
  --reload_codes)
    RELOAD_CODES="$2"
    shift 2
    ;;
  --reload_vocab)
    RELOAD_VOCAB="$2"
    shift 2
    ;;
  --product)
    PRDT="$2"
    shift 2
    ;;
  *)
    POSITIONAL+=("$1")
    shift
    ;;
  esac
done
set -- "${POSITIONAL[@]}"

#
# Check parameters
#
if [ "${SRC}" == "" ]; then
  echo "--src not provided"
  exit
fi
if [ "${TGT}" == "" ]; then
  echo "--tgt not provided"
  exit
fi
if [ "${SRC}" == "${TGT}" ]; then
  echo "source and target cannot be identical"
  exit
fi
# if [ "${SRC}" \> "${TGT}" ]; then echo "please ensure {SRC} < {TGT}"; exit; fi
if [ "$RELOAD_CODES" != "" ] && [ ! -f "$RELOAD_CODES" ]; then
  echo "cannot locate BPE codes"
  exit
fi
if [ "$RELOAD_VOCAB" != "" ] && [ ! -f "$RELOAD_VOCAB" ]; then
  echo "cannot locate vocabulary"
  exit
fi
if [ "$RELOAD_CODES" == "" -a "$RELOAD_VOCAB" != "" -o "$RELOAD_CODES" != "" -a "$RELOAD_VOCAB" == "" ]; then
  echo "BPE codes should be provided if and only if vocabulary is also provided"
  exit
fi

#
# Initialize tools and data paths
#

# main paths
MAIN_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
MONO_PATH=$DATA_PATH/mono

PROC_PATH=$DATA_PATH/processed/${SRC}-${TGT}
if [ "${SRC}" \> "${TGT}" ]; then PROC_PATH=$DATA_PATH/processed/${TGT}-${SRC}; fi
# PROC_PATH=$DATA_PATH/processed-vanilla-finetune-xlm17/${SRC}-${TGT}
# if [ "${SRC}" \> "${TGT}" ]; then PROC_PATH=$DATA_PATH/processed-vanilla-finetune-xlm17/${TGT}-${SRC}; fi

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $DATA_PATH
mkdir -p $MONO_PATH
mkdir -p $PROC_PATH

# moses
MOSES=$TOOLS_PATH/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$TOOLS_PATH/fastBPE/fast

# Sennrich's WMT16 scripts for Romanian preprocessing
WMT16_SCRIPTS=$TOOLS_PATH/wmt16-scripts
NORMALIZE_ROMANIAN=$WMT16_SCRIPTS/preprocess/normalise-romanian.py
REMOVE_DIACRITICS=$WMT16_SCRIPTS/preprocess/remove-diacritics.py

# raw input and tokenized files
SRC_TRAIN_RAW=$MONO_PATH/${SRC}/train_${SRC}_${PRDT}_x.txt
SRC_VALID_RAW=$MONO_PATH/${SRC}/valid_${SRC}_${PRDT}_x.txt
TGT_TEST_RAW=$MONO_PATH/${TGT}/test_${TGT}_${PRDT}_x.txt
# tokenized files from raw input
SRC_TRAIN_TOK=$SRC_TRAIN_RAW.tok
SRC_VALID_TOK=$SRC_VALID_RAW.tok
TGT_TEST_TOK=$TGT_TEST_RAW.tok

# BPE / vocab files
BPE_CODES=$PROC_PATH/codes
FULL_VOCAB=$PROC_PATH/vocab.${SRC}-${TGT}
if [ "${SRC}" \> "${TGT}" ]; then FULL_VOCAB=$PROC_PATH/vocab.${TGT}-${SRC}; fi

# train / valid / test monolingual BPE data:
# BPE data to be generated
SRC_TRAIN_BPE=$PROC_PATH/train_${SRC}_${PRDT}_x.bpe
SRC_VALID_BPE=$PROC_PATH/valid_${SRC}_${PRDT}_x.bpe
TGT_TEST_BPE=$PROC_PATH/test_${TGT}_${PRDT}_x.bpe

# install tools
# ./install-tools.sh

cd $MONO_PATH

# preprocessing commands - special case for Romanian, Chinese and Japanese
if [ "$SRC" == "ro" ]; then
  SRC_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -l $SRC -no-escape -threads $N_THREADS"
else
  SRC_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR |        $TOKENIZER -l $SRC -no-escape -threads $N_THREADS"
fi

if [ "$TGT" == "ro" ]; then
  TGT_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $TGT | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -l $TGT -no-escape -threads $N_THREADS"
elif [ "$TGT" == "jp" ]; then
  tmpTGT='ja'
  TGT_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $tmpTGT | $REM_NON_PRINT_CHAR | kytea -notags"
elif [ "$TGT" == "zh" ]; then
  TGT_PREPROCESSING="$TOOLS_PATH/stanford-segmenter-*/segment.sh pku /dev/stdin UTF-8 0 | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $TGT | $REM_NON_PRINT_CHAR"
else
  TGT_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $TGT | $REM_NON_PRINT_CHAR |       $TOKENIZER -l $TGT -no-escape -threads $N_THREADS"
fi

# tokenize data
# {SRC}_TRAIN_TOK=${SRC}_TRAIN_RAW.tok
# {SRC}_VALID_TOK=${SRC}_VALID_RAW.tok
# {TGT}_TEST_TOK=${TGT}_TEST_RAW.tok
if ! [[ -f "$SRC_TRAIN_TOK" ]]; then
  echo "Tokenize ${SRC} monolingual CLF training data..."
  eval "cat $SRC_TRAIN_RAW | $SRC_PREPROCESSING > $SRC_TRAIN_TOK"
fi

if ! [[ -f "$SRC_VALID_TOK" ]]; then
  echo "Tokenize $SRC monolingual CLF validation data..."
  eval "cat $SRC_VALID_RAW | $SRC_PREPROCESSING > $SRC_VALID_TOK"
fi

if ! [[ -f "$TGT_TEST_TOK" ]]; then
  echo "Tokenize ${TGT} monolingual CLF test data..."
  eval "cat $TGT_TEST_RAW | $TGT_PREPROCESSING > $TGT_TEST_TOK"
fi
# source lang mono langual raw: ZH, AR use yelp EN data.
# DE, FR, JP used unlabeled review data as monolingual unlabeled data.
# SRC_MONO_RAW=$MONO_PATH/${SRC}/train_${SRC}_hotel_x.txt
SRC_MONO_RAW=$MONO_PATH/${SRC}/unlabeled_${SRC}_x.txt
# tokenized files from raw input
SRC_MONO_TOK=$SRC_MONO_RAW.tok
if ! [[ -f "$SRC_MONO_TOK" ]]; then
  echo "Tokenize ${SRC} monolingual data..."
  eval "cat $SRC_MONO_RAW | $SRC_PREPROCESSING > $SRC_MONO_TOK"
fi
# target lang mono langual raw
TGT_MONO_RAW=$MONO_PATH/${TGT}/unlabeled_${TGT}_x.txt
# tokenized files from raw input
TGT_MONO_TOK=$TGT_MONO_RAW.tok
if ! [[ -f "$TGT_MONO_TOK" ]]; then
  echo "Tokenize ${TGT} monolingual data..."
  eval "cat $TGT_MONO_RAW | $TGT_PREPROCESSING > $TGT_MONO_TOK"
fi

echo "${SRC} monolingual CLF training data tokenized in: $SRC_TRAIN_TOK"
echo "${SRC} monolingual CLF validation data tokenized in: $SRC_VALID_TOK"
echo "${SRC} monolingual unlabeled data tokenized in: $SRC_MONO_TOK"
echo "${TGT} monolingual CLF test data tokenized in: $TGT_TEST_TOK"
echo "${TGT} monolingual unlabeled data tokenized in: $TGT_MONO_TOK"

# must reload existing BPE codes to BPE classification data
cd $MAIN_PATH
if [ ! -f "$BPE_CODES" ] && [ -f "$RELOAD_CODES" ]; then
  echo "Reloading BPE codes from $RELOAD_CODES ..."
  cp $RELOAD_CODES $BPE_CODES
fi

echo "BPE reloaded in $BPE_CODES"

# apply BPE codes to train, valid, test data
if ! [[ -f "$SRC_TRAIN_BPE" ]]; then
  echo "Applying BPE codes to ${SRC} train ..."
  $FASTBPE applybpe $SRC_TRAIN_BPE $SRC_TRAIN_TOK $BPE_CODES
fi

if ! [[ -f "$SRC_VALID_BPE" ]]; then
  echo "Applying BPE codes to ${SRC} valid ..."
  $FASTBPE applybpe $SRC_VALID_BPE $SRC_VALID_TOK $BPE_CODES
fi

if ! [[ -f "$TGT_TEST_BPE" ]]; then
  echo "Applying BPE codes to ${TGT} test ..."
  $FASTBPE applybpe $TGT_TEST_BPE $TGT_TEST_TOK $BPE_CODES
fi

SRC_MONO_BPE=$PROC_PATH/train.${SRC}
if ! [[ -f "$SRC_MONO_BPE" ]]; then
  echo "Applying BPE codes to ${SRC} monolingual unlabeled data ..."
  $FASTBPE applybpe $SRC_MONO_BPE $SRC_MONO_TOK $BPE_CODES
fi

TGT_MONO_BPE=$PROC_PATH/train.${TGT}
if ! [[ -f "$TGT_MONO_BPE" ]]; then
  echo "Applying BPE codes to ${TGT} monolingual unlabeled data ..."
  $FASTBPE applybpe $TGT_MONO_BPE $TGT_MONO_TOK $BPE_CODES
fi

echo "BPE codes applied to ${SRC} train in: $SRC_TRAIN_BPE"
echo "BPE codes applied to ${SRC} valid in: $SRC_VALID_BPE"
echo "BPE codes applied to ${SRC} mono in: $SRC_MONO_BPE"
echo "BPE codes applied to ${TGT} test in: $TGT_TEST_BPE"
echo "BPE codes applied to ${TGT} mono in: $TGT_MONO_BPE"

# reload pretrained full vocabulary to working folder
# must reload pretrained vocab to process classification data
cd $MAIN_PATH
if [ ! -f "$FULL_VOCAB" ] && [ -f "$RELOAD_VOCAB" ]; then
  echo "Reloading vocabulary from $RELOAD_VOCAB ..."
  cp $RELOAD_VOCAB $FULL_VOCAB
fi
echo "Full vocab in: $FULL_VOCAB"

# process unlabeled AR/ZH data (English has been processed already yelp 650k)
# First tokenize them
# Apply BPE

# binarize SRC monolingual training data
if ! [[ -f "$SRC_MONO_BPE.pth" ]]; then
  echo "Binarizing ${SRC} MONO unlabeled data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $SRC_MONO_BPE &
fi

# binarize TGT monolingual training data
if ! [[ -f "$TGT_MONO_BPE.pth" ]]; then
  echo "Binarizing ${TGT} MONO unlabeled data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TGT_MONO_BPE &
fi

# binarize train bpe data
if ! [[ -f "$SRC_TRAIN_BPE.pth" ]]; then
  echo "Binarizing ${SRC} train data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $SRC_TRAIN_BPE &
fi

if ! [[ -f "$SRC_VALID_BPE.pth" ]]; then
  echo "Binarizing ${SRC} valid data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $SRC_VALID_BPE &
fi

if ! [[ -f "$TGT_TEST_BPE.pth" ]]; then
  echo "Binarizing ${TGT} test data..."
  $MAIN_PATH/preprocess.py $FULL_VOCAB $TGT_TEST_BPE &
fi

echo "${SRC} binarized train data in: $SRC_TRAIN_BPE.pth"
echo "${SRC} binarized valid data in: $SRC_VALID_BPE.pth"
echo "${SRC} binarized mono data in: $SRC_MONO_BPE.pth"
echo "${TGT} binarized test data in: $TGT_TEST_BPE.pth"
echo "${TGT} binarized mono data in: $TGT_MONO_BPE.pth"
#
# Summary
#
echo ""
echo "===== Data summary"
echo "Monolingual training data:"
echo "    ${SRC}: $SRC_TRAIN_BPE.pth"
echo "    ${SRC}: $SRC_TRAIN_MONO_BPE.pth"
echo "Monolingual validation data:"
echo "    ${SRC}: $SRC_VALID_BPE.pth"
echo "Monolingual test data:"
echo "    ${TGT}: $TGT_TEST_BPE.pth"
echo "Monolingual training unlabeled data:"
echo "    ${SRC}: $SRC_MONO_BPE.pth"
echo "    ${TGT}: $TGT_MONO_BPE.pth"
echo ""
