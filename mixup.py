import cv2
import torch
import numpy as np


def mixup_data(x1,x2, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # batch_size = x.size()[0]
    # if use_cuda:
    #     index = torch.randperm(batch_size).cuda()
    # else:
    #     index = torch.randperm(batch_size)

    mixed_x = lam * x1 + (1 - lam) * x2
    # y_a, y_b = y, y[index]
    return mixed_x, lam


if __name__ == "__main__":

    img = cv2.imread('/root/Storage/datasets/total_text/train/train_images/img101.jpg')
    img2 = cv2.imread('/root/Storage/datasets/total_text/train/train_images/img1010.jpg')

    img = cv2.resize(img, (640, 640))
    img2 = cv2.resize(img2, (640, 640))

    mixed_x, lam = mixup_data(img,img2,100)

    cv2.imwrite('img.jpg', img)
    cv2.imwrite('mixed.jpg', mixed_x)