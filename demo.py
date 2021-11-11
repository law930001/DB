#!python3
import argparse
import os
import torch
import cv2
import numpy as np
from experiment import Structure, Experiment
from concern.config import Configurable, Config
import math
import natsort

def main():
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('exp', type=str)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--image_path', type=str, help='image path')
    parser.add_argument('--result_dir', type=str, default='./demo_results/', help='path to save results')
    parser.add_argument('--data', type=str,
                        help='The name of dataloader which will be evaluated on.')
    parser.add_argument('--image_short_side', type=int, default=736,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--thresh', type=float,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--box_thresh', type=float, default=0.6,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard')
    parser.add_argument('--resize', action='store_true',
                        help='resize')
    parser.add_argument('--polygon', action='store_true',
                        help='output polygons if true')
    parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
                        help='Show iamges eagerly')

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    Demo(experiment, experiment_args, cmd=args).inference(args['image_path'], args['visualize'])


class Demo:
    def __init__(self, experiment, args, cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.args = cmd
        model_saver = experiment.train.model_saver
        self.structure = experiment.structure
        self.model_path = self.args['resume']

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        model.load_state_dict(states, strict=False)

        # torch.save(model.state_dict(), 'model_best', _use_new_zipfile_serialization=False)

        print("Resumed from " + path)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        # resized_img = cv2.resize(img, (new_width, new_height))
        resized_img = cv2.resize(img, (2048, 1152))
        return resized_img
        
    def load_image(self, mask_img):
        # img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        img = mask_img.astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape
        
    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i,:,:].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")
        
    def inference(self, image_path, visualize=False):

        TBrain_root = '/root/Storage/datasets/TBrain/public/img_public/'
        csv_root = '/root/Storage/datasets/TBrain/public/Task2_Public_String_Coordinate.csv'

        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)
        model.eval()

        with open(csv_root, 'r') as csv_file:

            for i, line in enumerate(csv_file.readlines()):

                print(i + 1)

                line_split = line.strip().split(',')

                img_name = line_split[0]

                img_root = TBrain_root + img_name + '.jpg'

                ori_img = cv2.imread(img_root)

                x_max = max(int(line_split[1]),int(line_split[3]),int(line_split[5]),int(line_split[7]))
                y_max = max(int(line_split[2]),int(line_split[4]),int(line_split[6]),int(line_split[8]))
                x_min = min(int(line_split[1]),int(line_split[3]),int(line_split[5]),int(line_split[7]))
                y_min = min(int(line_split[2]),int(line_split[4]),int(line_split[6]),int(line_split[8]))

                # crop image
                # cropped = img[y_min:y_max, x_min:x_max]

                # cv2.imwrite('cropped.jpg', cropped)
                cv2.imwrite('img.jpg', ori_img)

                mask = np.zeros(ori_img.shape)
                mask[y_min:y_max, x_min:x_max] = 1

                mask_img = ori_img * mask

                # cv2.imwrite('mask_img.jpg', mask_img)


                batch = dict()
                batch['filename'] = [img_root]
                img, original_shape = self.load_image(mask_img)
                batch['shape'] = [original_shape]
                with torch.no_grad():
                    batch['image'] = img
                    pred = model.forward(batch, training=False)
                    output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon']) 
                    if not os.path.isdir(self.args['result_dir']):
                        os.mkdir(self.args['result_dir'])
                    self.format_output(batch, output)
                    temp = output[0][0]
                    crop_list = []
                    for line in temp:
                        line = np.array(line)
                        x_max = max(line[:,0])
                        y_max = max(line[:,1])
                        x_min = min(line[:,0])
                        y_min = min(line[:,1])

                        crop_list.append([x_max,y_max,x_min,y_min])

                    for j, cp in enumerate(sorted(crop_list,key=lambda l:(l[2]+l[3]))):

                        temp = ori_img[cp[3]:cp[1], cp[2]:cp[0]]
                        cv2.imwrite('temp/' + str(i+1) + '_' + str(j+1) + '.jpg', temp)

                    # if visualize and self.structure.visualizer:
                    #     vis_image = self.structure.visualizer.demo_visualize('./img.jpg', output)
                    #     cv2.imwrite(os.path.join(self.args['result_dir'], img_root.split('/')[-1].split('.')[0]+'.jpg'), vis_image)

if __name__ == '__main__':
    main()
