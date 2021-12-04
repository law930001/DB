cd ../
CUDA_VISIBLE_DEVICES=0 \
python train.py experiments/seg_detector/tt_resnet50_deform_thre.yaml \
--num_gpus 1 \
--num_workers 10 \
--batch_size 3
# --resume /root/Storage/DB_v100/workspace/SegDetectorModel-seg_detector/efficentnet_b7/L1BalanceCELoss/model/model_epoch_414_minibatch_138000 \
# --start_iter 138000 \
# --start_epoch 414