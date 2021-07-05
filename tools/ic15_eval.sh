cd ../
CUDA_VISIBLE_DEVICES=2 \
python eval.py experiments/seg_detector/ic15_resnet50_deform_thre.yaml \
--resume /root/Storage/DB/workspace/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/model_epoch_1200 \
--polygon \
--box_thresh 0.7
