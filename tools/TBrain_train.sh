cd ../
CUDA_VISIBLE_DEVICES=0 \
python train.py experiments/seg_detector/TBrain_resnet152_deform_thre.yaml \
--num_gpus 1 \
--num_workers 10 \
--batch_size 13 \
--resume workspace/SegDetectorModel-seg_detector/deformable_resnet152/L1BalanceCELoss/model/model_epoch_1_minibatch_129000