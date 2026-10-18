"""OSTD detector: Cascade R-CNN + DetectoRS RFP backbone/neck, DRFC modules,
FBDA attention on the finest pyramid feature, and Laplacian boundary
supervision during training.

The detector, the FBDA (dual-stream ASPP attention), the boundary head, and the
class-balanced boundary loss are derived from BAFNet (MIT, (c) 2022 Chang Xu);
see THIRD_PARTY_NOTICES.md. The DRFC modules and scale-balanced label
assignment are this repository's contributions.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from ..builder import DETECTORS
from ..losses.boundary_loss import LaplacianBoundaryLoss
from ..losses.router_balance_loss import RouterBalanceLoss
from ..boundary_head import boundary_head
from ..utils.drfc import DRFC
from .two_stage import TwoStageDetector


class PredLayer(nn.Module):
    """3x3+BN+ReLU 编码后接 1x1 投影, 输出中间特征与单通道 logits."""

    def __init__(self, in_c=256):
        super().__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_c, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        x = self.enlayer(x)
        return x, self.outlayer(x)


class ASPP(nn.Module):
    """1x1 与 dilation 3/5/7 的 3x3 四分支, 每支 256 通道."""

    def __init__(self, in_c):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_c, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_c, 256, 3, 1, padding=3, dilation=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_c, 256, 3, 1, padding=5, dilation=5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_c, 256, 3, 1, padding=7, dilation=7),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        return torch.cat((x1, x2, x3, x4), dim=1)


class FBDA(nn.Module):
    """Foreground-Background Decoupled Attention.

    用粗层 $P_4$ 产生单通道引导 logits $S_4$, 双线性上采样后取 sigmoid 得到
    注意力图 $A$；$F_a \\odot A$ 为前景流, $F_a \\odot (1-A)$ 为背景流。两条流
    各过独立的四分支 ASPP, 拼接后由 3x3 降回 256 通道得到 $F_d$。
    """

    def __init__(self, in_c):
        super(FBDA, self).__init__()
        self.ff_conv = ASPP(in_c)
        self.bf_conv = ASPP(in_c)
        self.rgbd_pred_layer = PredLayer(256 * 8)

    def forward(self, feat, pred):
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(
                pred, size=(H, W), mode='bilinear', align_corners=True))
        ff_feat = self.ff_conv(feat * pred)
        bf_feat = self.bf_conv(feat * (1 - pred))
        enhanced_feat, new_pred = self.rgbd_pred_layer(
            torch.cat((ff_feat, bf_feat), 1))
        return enhanced_feat, new_pred


@DETECTORS.register_module()
class OSTDDetector(TwoStageDetector):
    """Cascade R-CNN with DetectoRS Recursive Feature Pyramid, DRFC modules at
    the input stem / Layer2-to-Layer3 transition / finest pyramid feature,
    FBDA refinement, and Laplacian boundary supervision during training."""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 drfc_compatible_init=False,
                 p0_residual_scale=None,
                 p0_drfc_enabled=True):
        super(OSTDDetector, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.boundary_loss_func = LaplacianBoundaryLoss()
        self.coarse_pred = PredLayer(256)
        self.fbda = FBDA(256)
        self.boundary_head = boundary_head(256, 256, 1)
        self.p0_drfc = (DRFC(in_c=256, out_c=256, lightweight=True, top_k=3)
                        if p0_drfc_enabled else None)
        self.router_balance_loss = RouterBalanceLoss(
            num_experts=4, alpha=0.1, beta=0.01)
        self.drfc_compatible_init = drfc_compatible_init
        self.p0_residual_scale = p0_residual_scale

    def init_weights(self):
        super().init_weights()
        if (not self.drfc_compatible_init
                or getattr(self, '_drfc_compat_initialized', False)):
            return
        # 在 stem 与 Layer3 位置复用原 ImageNet 卷积作为 DRFC 快捷路径，
        # 使初始化时最接近原骨干；新增专家、路由器与 LCE 保持随机初始化。
        for backbone in [self.backbone, self.neck.rfp_modules[0]]:
            stem = backbone.drfc_stem
            layer3 = backbone.drfc_layer3
            if stem is not None:
                stem.shortcut = nn.Sequential(copy.deepcopy(backbone.conv1),
                                              copy.deepcopy(backbone.norm1))
            if layer3 is not None:
                layer3.shortcut = copy.deepcopy(backbone.layer3[0].downsample)
            for shortcut in [module.shortcut for module in (stem, layer3)
                             if module is not None]:
                for module in shortcut.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module._drfc_pretrained_bn = True
            # 非零 BN 缩放保证新专家路径仍有梯度。
            with torch.no_grad():
                if stem is not None:
                    stem.bn.weight.fill_(0.01)
                if layer3 is not None:
                    layer3.bn.weight.fill_(0.01)
        if self.p0_drfc is not None:
            with torch.no_grad():
                self.p0_drfc.bn.weight.fill_(0.01)
        self._drfc_compat_initialized = True

    def extract_feat(self, img, img_metas):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        x1 = list(x)
        # 由最粗层 P_4 产生 FBDA 的引导 logits。
        _, p4 = self.coarse_pred(x1[4])
        # 最细层 P_0 的轻量 DRFC 精化 (残差缩放)。
        if self.p0_drfc is None:
            p0_refined = x1[0]
        else:
            p0_output = self.p0_drfc(x1[0])
            if self.p0_residual_scale is None:
                p0_refined = p0_output
            else:
                p0_refined = x1[0] + self.p0_residual_scale * (
                    p0_output - x1[0])
        ef, _p = self.fbda(p0_refined, p4)
        if self.p0_residual_scale is None:
            x1[0] = ef + x1[0] + p0_refined
        else:
            x1[0] = ef + p0_refined
        return x1

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        x = self.extract_feat(img, img_metas)
        # 共享边界头在 P_0' 与 P_4 上各预测一次 (训练期使用)。
        x_b0 = self.boundary_head(x[0])
        x_b4 = self.boundary_head(x[4])
        losses = dict()
        # 边界监督的目标来自输入图像的灰度版本, 不需要分割掩码。
        gray = transforms.Grayscale()(torch.squeeze(img, 1))
        boundary_loss = (self.boundary_loss_func(x_b0, gray) +
                         self.boundary_loss_func(x_b4, gray))

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        losses['loss_boundary'] = boundary_loss

        # DRFC 路由正则: 遍历所有激活的 DRFC 模块, 累加其最近一次的
        # 路由器 logits 上的均衡损失。
        aux_loss = img.new_zeros(())
        for module in self.modules():
            if isinstance(module, DRFC) and module._last_router_logits is not None:
                aux_loss = aux_loss + self.router_balance_loss(
                    module._last_router_logits)
        losses['loss_router'] = aux_loss
        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img, img_metas)
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
            result (Tensor or tuple): The results to draw on `img`
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
        return super(OSTDDetector, self).show_result(data, result, **kwargs)
