cd ../
python demo.py \
experiments/seg_detector/TBrain_efficient_thre.yaml \
--image_path /root/Storage/datasets/TBrain/train/train_images/img_14.jpg \
--resume /root/Storage/DB_v100/workspace/SegDetectorModel-seg_detector/efficentnet_b7/L1BalanceCELoss/model/model_epoch_721_minibatch_1098000 \
--polygon \
--box_thresh 0.7 \
--visualize