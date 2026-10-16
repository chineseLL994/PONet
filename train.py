coding = 'utf-8'
import os
from net import Mnet
# from net import HAIMNet_VGG
from tqdm import tqdm
import random
import numpy as np
from torch.utils.data import DataLoader
from lib.dataset import Data
from lib.data_prefetcher import DataPrefetcher
from smooth_loss import get_saliency_smoothness
# from lib.loss import SSIM
import pytorch_iou
import torch
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
IOU = pytorch_iou.IOU(size_average = True)

def ssim_loss(pred, target, window_size=11, size_average=True):
    # Adapted from CVPR 2023 "Structure-Aware Saliency Detection"
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, window_size, 1, window_size // 2, count_include_pad=False)
    mu_y = F.avg_pool2d(target, window_size, 1, window_size // 2, count_include_pad=False)

    sigma_x = F.avg_pool2d(pred ** 2, window_size, 1, window_size // 2, count_include_pad=False) - mu_x ** 2
    sigma_y = F.avg_pool2d(target ** 2, window_size, 1, window_size // 2, count_include_pad=False) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, window_size, 1, window_size // 2, count_include_pad=False) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
                (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map


def edge_aware_loss(pred, target, edge_ratio=0.3):
    # From CVPR 2022 "Edge-Guided Saliency Detection"
    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device).view(1,
                                                                                                                      1,
                                                                                                                      3,
                                                                                                                      3)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device).view(1,
                                                                                                                      1,
                                                                                                                      3,
                                                                                                                      3)

    edge_x = F.conv2d(target, sobel_kernel_x, padding=1)
    edge_y = F.conv2d(target, sobel_kernel_y, padding=1)
    edge_map = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    edge_mask = (edge_map > edge_ratio).float()

    non_edge_mask = 1 - edge_mask
    edge_loss = F.binary_cross_entropy(pred * edge_mask, target * edge_mask, reduction='mean')
    non_edge_loss = F.binary_cross_entropy(pred * non_edge_mask, target * non_edge_mask, reduction='mean')

    return 0.7 * edge_loss + 0.3 * non_edge_loss


def multi_scale_consistency_loss(preds, target, scales=[0.5, 1.0, 2.0]):
    loss = 0.0
    for scale in scales:
        if scale != 1.0:
            scaled_pred = F.interpolate(preds, scale_factor=scale, mode='bilinear', align_corners=False)
            scaled_target = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
        else:
            scaled_pred = preds
            scaled_target = target

        # Combine SSIM and BCE at each scale
        loss += 0.5 * F.binary_cross_entropy(scaled_pred, scaled_target) + \
                0.5 * ssim_loss(scaled_pred, scaled_target)

    return loss / len(scales)


def my_loss1(score1, score2, score3, score4, s1_sig, s2_sig, s3_sig, s4_sig, label):
    # Basic losses
    sal_loss1 = F.binary_cross_entropy_with_logits(score1, label, reduction='mean')
    sal_loss2 = F.binary_cross_entropy_with_logits(score2, label, reduction='mean')
    sal_loss3 = F.binary_cross_entropy_with_logits(score3, label, reduction='mean')
    sal_loss4 = F.binary_cross_entropy_with_logits(score4, label, reduction='mean')

    # Smoothness loss
    sml = get_saliency_smoothness(torch.sigmoid(score1), label)

    # IOU losses
    iou_loss1 = IOU(s1_sig, label)
    iou_loss2 = IOU(s2_sig, label)
    iou_loss3 = IOU(s3_sig, label)
    iou_loss4 = IOU(s4_sig, label)

    # Advanced losses (CVPR 2022-2023)
    ssim_loss1 = ssim_loss(s1_sig, label)
    edge_loss1 = edge_aware_loss(s1_sig, label)
    ms_loss1 = multi_scale_consistency_loss(s1_sig, label)

    # Combine all losses with carefully tuned weights
    loss1 = sal_loss1 + iou_loss1 + 0.5 * ssim_loss1 + 0.3 * edge_loss1 + 0.2 * ms_loss1
    loss2 = sal_loss2 + iou_loss2
    loss3 = sal_loss3 + iou_loss3
    loss4 = sal_loss4 + iou_loss4

    # Final weighted combination
    total_loss = loss1 + loss2 + 0.9*loss3 + 0.8*loss4 + 0.8 * sml
    return total_loss
# def my_loss1(score1, score2, score3, score4, s1_sig, s2_sig, s3_sig, s4_sig, label):
#     sal_loss2 = F.binary_cross_entropy_with_logits(score2, label, reduction='mean')
#     sal_loss3 = F.binary_cross_entropy_with_logits(score3, label, reduction='mean')
#     sal_loss4 = F.binary_cross_entropy_with_logits(score4, label, reduction='mean')
#     sml = get_saliency_smoothness(torch.sigmoid(score1), label)
#     sal_loss1 = F.binary_cross_entropy_with_logits(score1, label, reduction='mean')
#     loss1 = sal_loss1 + IOU(s1_sig, label)
#     loss2 = sal_loss2 + IOU(s2_sig, label)
#     loss3 = sal_loss3 + IOU(s3_sig, label)
#     loss4 = sal_loss4 + IOU(s4_sig, label)
#     return loss1 + loss2 + 0.9 * loss3 + 0.8 * sml + 0.8 * loss4

# def my_loss1(score1, score2, score3, score4, score5, s1_sig, s2_sig, s3_sig, s4_sig, s5_sig, label):
#      sal_loss2 = F.binary_cross_entropy_with_logits(score2, label, reduction='mean')
#      sal_loss3 = F.binary_cross_entropy_with_logits(score3, label, reduction='mean')
#      sal_loss4 = F.binary_cross_entropy_with_logits(score4, label, reduction='mean')
#      sal_loss5 = F.binary_cross_entropy_with_logits(score5, label, reduction='mean')
#      sml = get_saliency_smoothness(torch.sigmoid(score1), label)
#      sal_loss1 = F.binary_cross_entropy_with_logits(score1, label, reduction='mean')
#      loss1 = sal_loss1 + loss_bce_ssim_iou(s1_sig, label)/3
#      loss2 = sal_loss2 + loss_bce_ssim_iou(s2_sig, label)/3
#      loss3 = sal_loss3 + loss_bce_ssim_iou(s3_sig, label)/3
#      loss4 = sal_loss4 + loss_bce_ssim_iou(s4_sig, label)/3
#      loss5 = sal_loss5 + loss_bce_ssim_iou(s5_sig, label)/3
#
#      return loss1 + loss2 + 0.9 * loss3 + 0.8 * sml + 0.8 * loss4 + 0.7 * loss5

if __name__ == '__main__':
    random.seed(825)
    np.random.seed(825)
    torch.manual_seed(825)
    torch.cuda.manual_seed(825)
    torch.cuda.manual_seed_all(825)

    # dataset
    img_root = 'D:/Chinese/HWSI/VDT-dataset/Train/'
    save_path = 'Revise/checkpoints/'
    # model_path = './model/epoch_20.pth'
    if not os.path.exists(save_path): os.mkdir(save_path)
    lr = 0.001
    batch_size = 4
    epoch = 400
    # lr_dec = [18, 34, 43, 53]
    num_params = 0
    data = Data(img_root)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers = 4)
    net = Mnet().cuda()
    net.load_pretrained_model()
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)
    params = net.parameters()
    optimizer = torch.optim.Adam(params, 0.0001, betas=(0.5, 0.999))


    for p in net.parameters():
        num_params += p.numel()
    print("The number of parameters: {}".format(num_params))
    iter_num = len(loader)
    # params = net.parameters()
    # optimizer = torch.optim.Adam(net.parameters(), 0.001)
    net.train()

    for epochi in tqdm(range(1, epoch + 1)):
        prefetcher = DataPrefetcher(loader)
        rgb, t, d, label= prefetcher.next()
        r_sal_loss = 0
        epoch_ave_loss = 0
        i = 0
        while rgb is not None:
            i += 1
            score1, score2, score3, score4, s1_sig, s2_sig, s3_sig, s4_sig, rgbdt_g, rgbdt4, rgbdt3, rgbdt2, rgbdt1 = net(rgb, t, d)
            sal_loss = my_loss1(score1, score2, score3, score4, s1_sig, s2_sig, s3_sig, s4_sig, label)
            r_sal_loss += sal_loss.data
            sal_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f, lr: %7.6f' % (
                    epochi, epoch, i, iter_num, r_sal_loss / 100, lr,))
                epoch_ave_loss += (r_sal_loss / 100)
                r_sal_loss = 0
            rgb, t, d, label = prefetcher.next()
        print('epoch-%2d_ave_loss: %7.6f' % (epochi, (epoch_ave_loss / (10.5 / batch_size))))
        if (epochi <= 350) and (epochi % 50 == 0):
            torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))
        elif (epochi > 350) and (epochi % 10 == 0):
            torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))

    torch.save(net.state_dict(), '%s/final.pth' % (save_path))

    # for epochi in tqdm(range(1, epoch + 1)):
    #     if epochi in lr_dec:
    #         lr = lr / 10
    #         optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005,momentum=0.9)
    #         print(lr)
    #     # optimizer = torch.optim.Adam(net.parameters(), 0.001)
    #     prefetcher = DataPrefetcher(loader)
    #     rgb, t, d, label = prefetcher.next()
    #     r_sal_loss = 0
    #     epoch_ave_loss = 0
    #     net.zero_grad()
    #     i = 0
    #     while rgb is not None:
    #         i += 1
    #         score1, score2, score3, score4, score5, s1_sig, s2_sig, s3_sig, s4_sig, s5_sig = net(rgb, t, d)
    #         sal_loss = my_loss1(score1, score2, score3, score4, score5, s1_sig, s2_sig, s3_sig, s4_sig, s5_sig, label)
    #         r_sal_loss += sal_loss.data
    #         sal_loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         if i % 100 == 0:
    #             print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f, lr: %7.6f' % (
    #                 epochi, epoch, i, iter_num, r_sal_loss / 100, lr,))
    #             epoch_ave_loss += (r_sal_loss / 100)
    #             r_sal_loss = 0
    #         rgb, t, d, label = prefetcher.next()
    #     print('epoch-%2d_ave_loss: %7.6f' % (epochi, (epoch_ave_loss / (10.5 / batch_size))))
    #     if (epochi <= 30) and (epochi % 10 == 0):
    #         torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))
    #     elif (epochi > 30) and (epochi % 1 == 0):
    #         torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))
    #
    # torch.save(net.state_dict(), '%s/final.pth' % (save_path))