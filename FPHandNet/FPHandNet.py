import torch
import torch.nn as nn
import torch.nn.functional as F
from FasterNet_Modified import FasterNet_T1
from util import ArcMarginProduct, CenterLoss,load_weight



class FPHandNet_FasterNet_T1(nn.Module):
    def __init__(self, num_classes=1000):
        super(FPHandNet_FasterNet_T1, self).__init__()
        # 构建指纹特征主干
        self.encoder = FasterNet_T1(num_classes=num_classes)
        self.encoder = load_weight(
            self.encoder,
            "/fasternet_t1-epoch.291-val_acc1.76.2180.pth"
        )

        # 构建掌纹特征主干
        self.encoder_palm = FasterNet_T1(num_classes=num_classes)
        self.encoder_palm = load_weight(
            self.encoder_palm,
            "/fasternet_t1-epoch.291-val_acc1.76.2180.pth"
        )

        self.GAP = nn.AdaptiveAvgPool2d((1, 1))

        # 分类器与 CenterLoss 配置
        # 其中 7680 和 1280 等数字是根据网络输出特征维度以及特征拼接方式而定
        self.classifier = ArcMarginProduct(7680, num_classes, m=1.0, easy_margin=True)
        self.criterion_cent = CenterLoss(num_classes=num_classes, feat_dim=1280)

        self.classifier_thumb = ArcMarginProduct(1280, num_classes, m=1.0, easy_margin=True)
        self.classifier_fore = ArcMarginProduct(1280, num_classes, m=1.0, easy_margin=True)
        self.classifier_middle = ArcMarginProduct(1280, num_classes, m=1.0, easy_margin=True)
        self.classifier_ring = ArcMarginProduct(1280, num_classes, m=1.0, easy_margin=True)
        self.classifier_little = ArcMarginProduct(1280, num_classes, m=1.0, easy_margin=True)
        self.classifier_palm = ArcMarginProduct(1280, num_classes, m=1.0, easy_margin=True)

        # 将所有手指（5路）特征与掌纹特征拼接后，再进行一次分类
        self.classifier_cat = ArcMarginProduct(1280 * 6, num_classes, m=1.0, easy_margin=True)

        self.criterion_cent_thumb = CenterLoss(num_classes=num_classes, feat_dim=1280)
        self.criterion_cent_fore = CenterLoss(num_classes=num_classes, feat_dim=1280)
        self.criterion_cent_middle = CenterLoss(num_classes=num_classes, feat_dim=1280)
        self.criterion_cent_ring = CenterLoss(num_classes=num_classes, feat_dim=1280)
        self.criterion_cent_little = CenterLoss(num_classes=num_classes, feat_dim=1280)
        self.criterion_cent_palm = CenterLoss(num_classes=num_classes, feat_dim=1280)
        self.criterion_cent_cat = CenterLoss(num_classes=num_classes, feat_dim=1280 * 6)

    def forward(self, x, label=None):
        """
        x: 期望是一个长度为2的元组或列表:
            x[0] -> 手指图像输入
            x[1] -> 掌纹图像输入
        label: 若不为 None，则进入训练/计算损失流程，否则只做前向推理
        """
        loss_tri = 0

        # 提取手指特征
        feature = self.encoder.extract_embedding(x[0])
        # 提取掌纹特征
        feature_palm = self.encoder_palm.extract_features(x[1])

        # 根据手指图像的切片规则，将同一张大图上的5根手指区域拆分出来
        feature_thumb = feature[:, :, 0:2, :]
        feature_fore = feature[:, :, 2:4, :]
        feature_middle = feature[:, :, 4:6, :]
        feature_ring = feature[:, :, 6:8, :]
        feature_little = feature[:, :, 8:10, :]

        # 分别通过预先定义好的下游操作（这里是 avgpool_pre_head_*）
        feature_thumb = self.encoder.avgpool_pre_head_thumb(feature_thumb)
        feature_thumb = torch.flatten(feature_thumb, 1)

        feature_fore = self.encoder.avgpool_pre_head_fore(feature_fore)
        feature_fore = torch.flatten(feature_fore, 1)

        feature_middle = self.encoder.avgpool_pre_head_middle(feature_middle)
        feature_middle = torch.flatten(feature_middle, 1)

        feature_ring = self.encoder.avgpool_pre_head_ring(feature_ring)
        feature_ring = torch.flatten(feature_ring, 1)

        feature_little = self.encoder.avgpool_pre_head_little(feature_little)
        feature_little = torch.flatten(feature_little, 1)

        # 将 5 根手指特征与掌纹特征拼接，形成更大的特征向量
        feature_cat = torch.cat([
            feature_thumb,
            feature_fore,
            feature_middle,
            feature_ring,
            feature_little,
            feature_palm
        ], dim=1)

        if label is not None:
            # 计算 5 路手指与掌纹的分类 logits
            logit_thumb = self.classifier_thumb(feature_thumb, label)
            loss_cent_thumb = self.criterion_cent_thumb(feature_thumb, label)

            logit_fore = self.classifier_fore(feature_fore, label)
            loss_cent_fore = self.criterion_cent_fore(feature_fore, label)

            logit_middle = self.classifier_middle(feature_middle, label)
            loss_cent_middle = self.criterion_cent_middle(feature_middle, label)

            logit_ring = self.classifier_ring(feature_ring, label)
            loss_cent_ring = self.criterion_cent_ring(feature_ring, label)

            logit_little = self.classifier_little(feature_little, label)
            loss_cent_little = self.criterion_cent_little(feature_little, label)

            logit_palm = self.classifier_palm(feature_palm, label)
            loss_cent_palm = self.criterion_cent_palm(feature_palm, label)

            return (
                feature_thumb,
                feature_fore,
                feature_middle,
                feature_ring,
                feature_little,
                feature_palm,
                feature_cat,
                logit_thumb,
                logit_fore,
                logit_middle,
                logit_ring,
                logit_little,
                logit_palm,
                loss_cent_thumb,
                loss_cent_fore,
                loss_cent_middle,
                loss_cent_ring,
                loss_cent_little,
                loss_cent_palm,
            )
        else:
            # 推理时仅返回特征
            return (feature_cat)
