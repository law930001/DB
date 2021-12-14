cd ../
python eval.py experiments/seg_detector/tt_efficient_thre.yaml \
--resume workspace/SegDetectorModel-seg_detector/efficentnet_b7/L1BalanceCELoss/model_exp22/final \
--polygon \
--box_thresh 0.6
