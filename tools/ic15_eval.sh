cd ../
python eval.py experiments/seg_detector/ic15_hrnet48_thre.yaml \
--resume model/model_epoch_3000_minibatch_375000 \
--polygon \
--box_thresh 0.6
