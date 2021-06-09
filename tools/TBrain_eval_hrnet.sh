cd ../
python eval.py experiments/seg_detector/TBrain_hrnet48_thre.yaml \
--resume /root/Storage/DB_v100/workspace/SegDetectorModel-seg_detector/hrnet48/L1BalanceCELoss/model/model_epoch_99_minibatch_198000 \
--polygon \
--box_thresh 0.6
