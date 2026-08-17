from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from ..losses.detail_loss import DetailAggregateLoss
from ..losses import LoadBalancingLoss
from ..model_utils import segmenthead
from ..utils import ARFC
import matplotlib.pyplot as plt
import numpy as np
BatchNorm2d = nn.BatchNorm2d
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.bn.train(True)
        self.bn.track_running_stats = False
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class Pred_Layer(nn.Module):
    def __init__(self, in_c=256):
        super(Pred_Layer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_c, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        x = self.enlayer(x)
        x1 = self.outlayer(x)
        return x, x1


class FeatureLaplacianGT(nn.Module):
    """特征域边界真值生成器 (方案 3.1).

    对 ARFC 精炼后的特征做 1x1 压缩 → 多尺度拉普拉斯提取 → 三尺度拼接。
    输出与原图域真值同形状的单通道二值张量，供
    DetailAggregateLoss.forward_with_gt 调用。

    Args:
        in_c (int): 输入特征通道数。
        thr (float): 二值化阈值，默认 0.1。
    """

    def __init__(self, in_c=256, thr=0.1):
        super().__init__()
        self.reduce = nn.Conv2d(in_c, 1, 1, bias=False)
        kernel = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1],
                              dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('lap', kernel)
        self.thr = thr

    def _laplacian(self, x, stride):
        return F.conv2d(x, self.lap, stride=stride, padding=1).clamp(min=0)

    def forward(self, feat):
        f = torch.sigmoid(self.reduce(feat))              # [B,1,H,W] 0~1
        f1 = self._laplacian(f, stride=1)
        f2 = self._laplacian(f, stride=2)
        f4 = self._laplacian(f, stride=4)
        # 上采样到 f1 的尺寸并二值化拼接
        h, w = f1.shape[-2:]
        f2_up = F.interpolate(f2, size=(h, w), mode='nearest')
        f4_up = F.interpolate(f4, size=(h, w), mode='nearest')
        f_stack = torch.cat([f1, f2_up, f4_up], dim=1)     # [B,3,H,W]
        # 借鉴 DetailAggregateLoss 的 fuse_kernel：可学习 1x1 conv 在此简化
        return f_stack                                    # 用 3 通道拼接代替 1 通道融合


class DualBoundaryGT(nn.Module):
    """原图域 + 特征域 联合边界 GT (方案 3.1 完整实现).

    输出 3 通道堆叠的真值图，分别对应 stride 1/2/4 拉普拉斯结果。
    """

    def __init__(self, feat_in_c=256):
        super().__init__()
        self.feat_gt = FeatureLaplacianGT(in_c=feat_in_c)
        # 原图域拉普拉斯核
        kernel = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1],
                              dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('lap', kernel)

    def _img_lap(self, img_gray):
        """img_gray: [B,1,H,W]."""
        g1 = F.conv2d(img_gray, self.lap, padding=1).clamp(min=0)
        g2 = F.interpolate(
            F.conv2d(img_gray, self.lap, stride=2, padding=1).clamp(min=0),
            g1.shape[-2:], mode='nearest')
        g4 = F.interpolate(
            F.conv2d(img_gray, self.lap, stride=4, padding=1).clamp(min=0),
            g1.shape[-2:], mode='nearest')
        return torch.cat([g1, g2, g4], dim=1)              # [B,3,H,W]

    def forward(self, img_gray, feat):
        img_stack = self._img_lap(img_gray)
        feat_stack = self.feat_gt(feat)
        target_h, target_w = img_stack.shape[-2:]
        if feat_stack.shape[-2:] != (target_h, target_w):
            feat_stack = F.interpolate(feat_stack,
                                       size=(target_h, target_w),
                                       mode='nearest')
        # 拼接为 [B, 6, H, W] 的联合真值
        return torch.cat([img_stack, feat_stack], dim=1)


# FF
class FF(nn.Module):
    def __init__(self, in_c):
        super(FF, self).__init__()
        self.reduce = nn.Conv2d(in_c, 32, 1)
        self.ff_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.rgbd_pred_layer = Pred_Layer(32)

    def forward(self, feat, pred):
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        ff_feat = self.ff_conv(feat * pred)
        enhanced_feat, new_pred = self.rgbd_pred_layer(ff_feat)
        return enhanced_feat, new_pred


# BF
class BF(nn.Module):
    def __init__(self, in_c):
        super(BF, self).__init__()
        self.reduce = nn.Conv2d(in_c * 2, 32, 1)
        self.bf_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.rgbd_pred_layer = Pred_Layer(32)

    def forward(self, feat, pred):
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        bf_feat = self.bf_conv(feat * (1 - pred))
        enhanced_feat, new_pred = self.rgbd_pred_layer(bf_feat)
        return enhanced_feat, new_pred


# ASPP for DSAM
class ASPP(nn.Module):
    def __init__(self, in_c):
        super(ASPP, self).__init__()

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_c , 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_c , 256, 3, 1, padding=3, dilation=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_c , 256, 3, 1, padding=5, dilation=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_c , 256, 3, 1, padding=7, dilation=7),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class DSAM(nn.Module):
    def  __init__(self, in_c):
        super(DSAM, self).__init__()
        self.ff_conv = ASPP(in_c)
        self.bf_conv = ASPP(in_c)
        self.rgbd_pred_layer = Pred_Layer(256 * 8)

    def forward(self, feat, pred):
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        ff_feat = self.ff_conv(feat * pred)
        bf_feat = self.bf_conv(feat * (1 - pred))
        enhanced_feat, new_pred = self.rgbd_pred_layer(torch.cat((ff_feat, bf_feat), 1))
        return enhanced_feat, new_pred
@DETECTORS.register_module()
class CascadeRCNN_BAF(TwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 use_arfc=True):
        super(CascadeRCNN_BAF, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.boundary_loss_func = DetailAggregateLoss()
        self.rgb_global = Pred_Layer(256)
        # === 改造 2：P0 前置轻量化 ARFC ===
        self.p0_arfc = ARFC(in_c=256, out_c=256,
                            lightweight=True, top_k=3)
        # === 改造 3：双路 GT + 多输入 seghead ===
        self.dual_gt = DualBoundaryGT(feat_in_c=256)
        self.seghead_dual = segmenthead(256 * 2, 256, 1)
        # === 均衡损失接入 ===
        self.aux_loss = LoadBalancingLoss(num_experts=4, alpha=0.1, beta=0.01)
        # === 原结构保留 ===
        self.dsam = DSAM(256)
        self.seghead = segmenthead(256,256,1)
        # ARFC 开关：基线训练时通过 cfg 注入 use_arfc=False 关掉 P0 ARFC，
        # 其余结构（boundary GT、LoadBalancingLoss、seghead_dual）保持不变。
        # 用于 ablate 实验区分 ARFC 在当前数据上的边际贡献。
        self.use_arfc = use_arfc

    def extract_feat(self, img,img_metas):
        #imgpath = img_metas[0]['filename']
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        x1 = list(x)
        #e3, p3 = self.rgb_global(x1[3])
        e4, p4 = self.rgb_global(x1[4])
        # [_, _, H, W] = p3.size()
        # p = F.interpolate(p4,
        #                   size=(H, W),
        #                   mode='bilinear',
        #                   align_corners=True) + p3
        # === 改造 2：P0 前置 ARFC（use_arfc=False 时退化为恒等映射） ===
        p0_refined = self.p0_arfc(x1[0]) if self.use_arfc else x1[0]
        ef, _p = self.dsam(p0_refined, p4)
        x1[0] = ef + x1[0] + p0_refined
        # === 改造 2 结束 ===
        return x1, p0_refined

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x, p0_refined = self.extract_feat(img, img_metas)

        # 灰度边界 GT: img[B,3,H,W] -> lb[B,1,H,W]
        transform = transforms.Grayscale()
        lb = transform(img)

        # === 改造 3：双路边界真值 ===
        fused_gt = self.dual_gt(lb, p0_refined)  # [B,6,H,W]
        # 改造 3.2：seghead 输入为 [S0, f_ARFC] 拼接
        seg_in = torch.cat([x[0], p0_refined], dim=1)
        x_b0 = self.seghead_dual(seg_in)
        x_b3 = self.seghead(x[4])
        # 使用 forward_with_gt 接受外部真值
        # fused_gt[:, :3] 是 img 侧 stride 1/2/4 三尺度拉普拉斯堆叠 (3ch,
        # 连续值)。forward_with_gt / weighted_bce 要求预测与 GT 通道一致
        # (x_b0 为 1ch)，因此先沿通道融合为 1ch 并按 0.1 阈值二值化，
        # 与 detail_loss.forward 原版逻辑保持一致。
        # P1-B：去二值化。原 >0.1 阈值把 ~99% 像素判定为背景，导致 BCE 梯度
        # 塌缩（模型预测全 0 即近最优）。改为连续 Laplacian / 8 后 clamp 到
        # [0, 1] 作为软标签，BCEWithLogits 能正常反传边缘像素的梯度。
        img_side_gt = fused_gt[:, :3, :, :].max(dim=1, keepdim=True)[0]
        img_side_gt = (img_side_gt / 8.0).clamp(0.0, 1.0)
        boundery_loss = (self.boundary_loss_func.forward_with_gt(
                             x_b0, img_side_gt)
                         + self.boundary_loss_func(
                             x_b3, lb))
        # === 改造 3 结束 ===

        # === 均衡损失 ===
        aux_loss = p0_refined.new_zeros(())
        for m in self.modules():
            if (isinstance(m, ARFC)
                    and getattr(m, '_last_router_logits', None) is not None):
                aux_loss = aux_loss + self.aux_loss(m._last_router_logits)
        # === 均衡损失结束 ===

        losses = dict()
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x, img_metas, gt_bboxes, gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            x, img_metas, proposal_list,
            gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks, **kwargs)
        losses.update(roi_losses)
        losses['loss_bounder'] = boundery_loss
        losses['loss_aux'] = aux_loss
        return losses
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x, _ = self.extract_feat(img, img_metas)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
    def show_result(self, data, result, **kwargs):
        """Show prediction results of the detector.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        return super(CascadeRCNN_BAF, self).show_result(data, result, **kwargs)

