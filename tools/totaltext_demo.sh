python demo.py \
experiments/seg_detector/totaltext_resnet50_deform_thre.yaml \
--image_path datasets/total_text/test_images/img10.jpg \
--resume model/totaltext_resnet50 \
--polygon \
--box_thresh 0.7 \
--visualize