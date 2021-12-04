import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

import backbones
import decoders

from torchsummary import summary



class BasicModel(nn.Module):
    def __init__(self, args):
        nn.Module.__init__(self)

        self.backbone = getattr(backbones, args['backbone'])(**args.get('backbone_args', {}))
        self.decoder = getattr(decoders, args['decoder'])(**args.get('decoder_args', {}))

        # summary(self.decoder.to('cuda'), [(256,160,160),(512,80,80),(1024,40,40),(2048,20,20)])


    def forward(self, data, *args, **kwargs):
        return self.decoder(self.backbone(data), *args, **kwargs)


def parallelize(model, distributed, local_rank):
    if distributed:
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=[local_rank],
            find_unused_parameters=True)
    else:
        return nn.DataParallel(model)

class SegDetectorModel(nn.Module):
    def __init__(self, args, device, distributed: bool = False, local_rank: int = 0):
        super(SegDetectorModel, self).__init__()
        from decoders.seg_detector_loss import SegDetectorLossBuilder

        self.model = BasicModel(args)
        # for loading models
        self.model = parallelize(self.model, distributed, local_rank)
        self.criterion = SegDetectorLossBuilder(
            args['loss_class'], *args.get('loss_args', []), **args.get('loss_kwargs', {})).build()
        self.criterion = parallelize(self.criterion, distributed, local_rank)
        self.device = device
        self.to(self.device)

    @staticmethod
    def model_name(args):
        return os.path.join('seg_detector', args['backbone'], args['loss_class'])

    def forward(self, batch, training=True):
        if isinstance(batch, dict):
            data = batch['image'].to(self.device)
        else:
            data = batch.to(self.device)
        data = data.float()

        if self.training:
            pred = self.model(data, training=self.training)
        else:
            # pred = self.model(data, training=self.training)
            pred, pred_whole = self.model(data, training=self.training)

        # file_num = batch['filename'][0].replace('img', '').replace('.jpg', '')
        # file_w, file_h = batch['shape'][0]

        # cv2.imwrite('results_heatmap_img/' + str(file_num) + '_binary.jpg', 
        #             cv2.resize(self.denormalize(pred, 0), (file_h,file_w)))
        # cv2.imwrite('results_heatmap_img/' + str(file_num) + '_whole_binary.jpg', 
        #             cv2.resize(self.denormalize(pred_whole, 0), (file_h,file_w)))


        if self.training:
            for key, value in batch.items():
                if value is not None:
                    if hasattr(value, 'to'):
                        batch[key] = value.to(self.device)
            loss_with_metrics = self.criterion(pred, batch)
            loss, metrics = loss_with_metrics
            return loss, pred, metrics
        return pred

    def denormalize(self, image, value):

        if len(image.shape) == 4:
            image = image[0]
            if value == 0:
                image = image > 0.3
            else:
                image = image > 0.3
            image = image.cpu().numpy()[0]
            bitmap = (image*255).astype(np.uint8)



        return bitmap