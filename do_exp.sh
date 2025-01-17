GPU=$1
DATASET=$2
DISTRIBUTION=$3
ALGORITHM=$4
ARCH=$5
AUGMENTATION=$6
OUTDIR=$7
LR=$8
ATTR=$9
PRETRAIN=${10}
DATADIR=${11}

if [ "$PRETRAIN" = "1" ]; then
       CUDA_VISIBLE_DEVICES=$GPU python -m domainbed.train \
              --data_dir=$DATADIR \
              --output_dir=$OUTDIR \
              --algorithm $ALGORITHM \
              --dataset $DATASET \
              --aug $AUGMENTATION \
              --dist_type $DISTRIBUTION\
              --arch $ARCH \
              --lr $LR \
              --attr $ATTR\
              --pretrain \

else
       CUDA_VISIBLE_DEVICES=$GPU python -m domainbed.train \
              --data_dir=$DATADIR \
              --output_dir=$OUTDIR \
              --algorithm $ALGORITHM \
              --dataset $DATASET \
              --aug $AUGMENTATION \
              --dist_type $DISTRIBUTION\
              --arch $ARCH \
              --lr $LR \
              --attr $ATTR\

fi