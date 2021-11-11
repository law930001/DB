cd ../
CUDA_VISIBLE_DEVICES=0 \
python train.py experiments/seg_detector/ic15_efficient_thre.yaml \
--num_gpus 1 \
--num_workers 10 \
--batch_size 3