cd ../
CUDA_VISIBLE_DEVICES=0 \
python train.py experiments/seg_detector/TBrain_hrnet48_thre.yaml \
--num_gpus 1 \
--num_workers 2 \
--batch_size 2 \
--start_iter 390000 \
--start_epoch 194 \
--resume /root/Storage/DB_v100/outputs/workspace/DB_v100/SegDetectorModel-seg_detector/hrnet48/L1BalanceCELoss/model/model_epoch_194_minibatch_390000