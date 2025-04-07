import cv2
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import os
from paddleseg import utils as u1
from paddleseg.models import layers
from paddleseg.cvlibs import manager
from paddleseg.utils import utils


def reverse_transform(pred, trans_info, mode='nearest'):
    """recover pred to origin shape"""
    intTypeList = [paddle.int8, paddle.int16, paddle.int32, paddle.int64]
    dtype = pred.dtype
    for item in trans_info[::-1]:
        if isinstance(item[0], list):
            trans_mode = item[0][0]
        else:
            trans_mode = item[0]
        if trans_mode == 'resize':
            h, w = item[1][0], item[1][1]
            if paddle.get_device() == 'cpu' and dtype in intTypeList:
                pred = paddle.cast(pred, 'float32')
                pred = F.interpolate(pred, [h, w], mode=mode)
                pred = paddle.cast(pred, dtype)
            else:
                pred = F.interpolate(pred, [h, w], mode=mode)
        elif trans_mode == 'padding':
            h, w = item[1][0], item[1][1]
            pred = pred[:, :, 0:h, 0:w]
        else:
            raise Exception("Unexpected info '{}' in im_info".format(item[0]))
    return pred

@manager.MODELS.add_component
class HSANet(nn.Layer):

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=[2, 3, 4],
                 arm_type='simpleUAFM_SpAtten',
                 cm_bin_sizes=[1, 2, 4],
                 cm_out_ch=128,
                 arm_out_chs=[64, 96, 128],
                 seg_head_inter_chs=[64, 64, 64],
                 resize_mode='bilinear',
                 pretrained=None):
        super().__init__()

        # backbone
        assert hasattr(backbone, 'feat_channels'), \
            "The backbone should has feat_channels."
        assert len(backbone.feat_channels) >= len(backbone_indices), \
            f"The length of input backbone_indices ({len(backbone_indices)}) should not be" \
            f"greater than the length of feat_channels ({len(backbone.feat_channels)})."
        assert len(backbone.feat_channels) > max(backbone_indices), \
            f"The max value ({max(backbone_indices)}) of backbone_indices should be " \
            f"less than the length of feat_channels ({len(backbone.feat_channels)})."
        self.backbone = backbone

        assert len(backbone_indices) > 1, "The lenght of backbone_indices " \
            "should be greater than 1"
        self.backbone_indices = backbone_indices  # [..., x16_id, x32_id]
        backbone_out_chs = [backbone.feat_channels[i] for i in backbone_indices]

        # head
        if len(arm_out_chs) == 1:
            arm_out_chs = arm_out_chs * len(backbone_indices)
        assert len(arm_out_chs) == len(backbone_indices), "The length of " \
            "arm_out_chs and backbone_indices should be equal"

        self.ppseg_head = HSANetHead(backbone_out_chs, arm_out_chs,
                                        cm_bin_sizes, cm_out_ch, arm_type,
                                        resize_mode)

        if len(seg_head_inter_chs) == 1:
            seg_head_inter_chs = seg_head_inter_chs * len(backbone_indices)
        assert len(seg_head_inter_chs) == len(backbone_indices), "The length of " \
            "seg_head_inter_chs and backbone_indices should be equal"
        self.seg_heads = nn.LayerList()  # [..., head_16, head32]
        for in_ch, mid_ch in zip(arm_out_chs, seg_head_inter_chs):
            self.seg_heads.append(SegHead(in_ch, mid_ch, num_classes))

        # pretrained
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]
        # trans_info=x['trans_info']
        feats_backbone = self.backbone(x)  # [x2, x4, x8, x16, x32]
        assert len(feats_backbone) >= len(self.backbone_indices), \
            f"The nums of backbone feats ({len(feats_backbone)}) should be greater or " \
            f"equal than the nums of backbone_indices ({len(self.backbone_indices)})"

        feats_selected = [feats_backbone[i] for i in self.backbone_indices]

        feats_head = self.ppseg_head(feats_selected)  # [..., x8, x16, x32]
 
        if self.training:
            x = self.seg_heads[0](feats_head[0])
            x = F.interpolate(x, x_hw, mode='bilinear', align_corners=False)
            logit_list = [x]
        else:
            logit_list = []

            for x, seg_head in zip(feats_head, self.seg_heads):
                x = seg_head(x)
                logit_list.append(x)

            logit_list = [
                F.interpolate(
                    x, x_hw, mode='bilinear', align_corners=False)
                for x in logit_list

            ]
            # for i in range(2):
            x = self.seg_heads[0](feats_head[0])
            x = F.interpolate(x, x_hw, mode='bilinear', align_corners=False)
            logit = reverse_transform(x, [], mode='bilinear')
            pred = paddle.argmax(x, axis=1, keepdim=True, dtype='int32')
            color_map = u1.visualize.get_color_map_list(256, custom_color=None)
            # for x in logit_list:
            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype('uint8')
            pred_mask = u1.visualize.get_pseudo_color_map(pred, color_map)
            

            x = self.seg_heads[1](feats_head[1])
            x = F.interpolate(x, x_hw, mode='bilinear', align_corners=False)
            logit = reverse_transform(x, [], mode='bilinear')
            pred = paddle.argmax(x, axis=1, keepdim=True, dtype='int32')
            color_map = u1.visualize.get_color_map_list(256, custom_color=None)
            # for x in logit_list:
            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype('uint8')
            pred_mask = u1.visualize.get_pseudo_color_map(pred, color_map)
            
            x = self.seg_heads[2](feats_head[2])
            x = F.interpolate(x, x_hw, mode='bilinear', align_corners=False)
            logit = reverse_transform(x, [], mode='bilinear')
            pred = paddle.argmax(x, axis=1, keepdim=True, dtype='int32')
            color_map = u1.visualize.get_color_map_list(256, custom_color=None)
            # for x in logit_list:
            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype('uint8')
            pred_mask = u1.visualize.get_pseudo_color_map(pred, color_map)
            

            x = self.seg_heads[0](feats_head[0])
            x = F.interpolate(x, x_hw, mode='bilinear', align_corners=False)
            logit_list = [x]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class HSANetHead(nn.Layer):
    """
    The head of HSANet.

    Args:
        backbone_out_chs (List(Tensor)): The channels of output tensors in the backbone.
        arm_out_chs (List(int)): The out channels of each arm module.
        cm_bin_sizes (List(int)): The bin size of context module.
        cm_out_ch (int): The output channel of the last context module.
        arm_type (str): The type of attention refinement module.
        resize_mode (str): The resize mode for the upsampling operation in decoder.
    """

    def __init__(self, backbone_out_chs, arm_out_chs, cm_bin_sizes, cm_out_ch,
                 arm_type, resize_mode):
        super().__init__()

        
        self.cm = SimpleConv(backbone_out_chs[-1] , cm_out_ch)
        assert hasattr(layers,arm_type), \
            "Not support arm_type ({})".format(arm_type)
        arm_class = eval("layers." + arm_type)

        self.arm_list = nn.LayerList()  # [..., arm8, arm16, arm32]
        for i in range(len(backbone_out_chs)):
            low_chs = backbone_out_chs[i]
            high_ch = cm_out_ch if i == len(
                backbone_out_chs) - 1 else arm_out_chs[i + 1]
            out_ch = arm_out_chs[i]
            arm = arm_class(
                low_chs, high_ch, out_ch, ksize=3, resize_mode=resize_mode)
            self.arm_list.append(arm)

    def forward(self, in_feat_list):
        """
        Args:
            in_feat_list (List(Tensor)): Such as [x2, x4, x8, x16, x32].
                x2, x4 and x8 are optional.
        Returns:
            out_feat_list (List(Tensor)): Such as [x2, x4, x8, x16, x32].
                x2, x4 and x8 are optional.
                The length of in_feat_list and out_feat_list are the same.
        """
        
        high_feat = self.cm(in_feat_list[-1])
        out_feat_list = []

        for i in reversed(range(len(in_feat_list))):
            low_feat = in_feat_list[i]
            arm = self.arm_list[i]
            high_feat = arm(low_feat, high_feat)
            out_feat_list.insert(0, high_feat)

        return out_feat_list

class SimpleConv(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ):
        super().__init__()
        self.conv1= layers.ConvBNReLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1)
    def forward(self, input):
            return self.conv1(input)



class SegHead(nn.Layer):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            in_chan,
            mid_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.conv_out = nn.Conv2D(
            mid_chan, n_classes, kernel_size=1, bias_attr=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x
