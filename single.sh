########### Zeroshot
# OUTDIR=../rob_exps/zeroshot
# MODE=Phi-3.5
# PROMPT=tailored

# DATASET=SHAPES3D
# python run_zeroshot.py --mode $MODE --output_dir $OUTDIR --prompt_type $PROMPT --dataset $DATASET &

########### 
GPU=0
DATASET=CELEBA       #DSPRITES, SHAPES3D, SMALLNORB, DEEPFASHION, CELEBA, iwildcam, FMOW, CAMELYON17
DISTRIBUTION=SC      #SC, LDD, UDS, SC_LDD, SC_UDS, LDD_UDS, SC_LDD_UDS
ALGORITHM=ERM        #ERM, ADA, ME_ADA, SagNet, L2D, IRM, CausIRL_MMD, CausIRL_CORAL, UBNet, PnD, GroupDRO, BPA
ARCH=resnet18        #resnet18, resnet50, resnet101, vit, mlp
AUG=no_aug  #no_aug,imgnet,augmix,randaug,autoaug
OUTDIR=../exps       #     
LR=0.0001            
ATTR=0               #[0,2] for single DS, [0,5] for concurrent DS
PRETRAIN=1           #0 for scratch, 1 for PRETRAIN
DATADIR=../NeurIPS2024        #directory where datasets are

bash do_exp.sh $GPU $DATASET $DISTRIBUTION $ALGORITHM $ARCH $AUG $OUTDIR $LR $ATTR $PRETRAIN $DATADIR &
