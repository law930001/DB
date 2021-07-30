cd ../
CUDA_VISIBLE_DEVICES=2 \
python train.py experiments/seg_detector/ic15_hrnet48_thre.yaml \
--num_gpus 1 \
--num_workers 10 \
--batch_size 24