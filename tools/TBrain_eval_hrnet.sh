cd ../
python eval.py experiments/seg_detector/TBrain_hrnet48_thre.yaml \
--resume model/hrnet/model_epoch_315_minibatch_126000 \
--polygon \
--box_thresh 0.6
