#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>
from models.submodules import *
from models.FAC.kernelconv2d import KernelConv2D

from torch import nn

from .blocks import ConvBlock, DeconvBlock, FeatureHeatmapFusingBlock, LinearWeightedAvg
from .stacked_hour_glass import FeedbackHourGlass, HourGlass
from .srfbn_hg_arch import FeedbackBlockCustom, FeedbackBlockHeatmapAttention, merge_heatmap_5
import utils.eval_landmarks as eval_landmarks

from .kfsgnet import KFSGNet

class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()
        #############################
        # Deblurring Branch
        #############################
        # encoder
        ks = 3
        ks_2d = 5
        ch1 = 32
        ch2 = 64
        ch3 = 128
        self.fea1 = conv(2*ch3, ch3, kernel_size=ks, stride=1)
        self.fea2 = conv(ch3, 2 * ch3, kernel_size=ks, stride=1)

        self.conv1_1 = conv(3, ch1, kernel_size=ks, stride=1)
        self.conv1_2 = resnet_block(ch1, kernel_size=ks)
        self.conv1_3 = resnet_block(ch1, kernel_size=ks)

        self.conv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.conv2_2 = resnet_block(ch2, kernel_size=ks)
        self.conv2_3 = resnet_block(ch2, kernel_size=ks)

        self.conv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.conv3_2 = resnet_block(ch3, kernel_size=ks)
        self.conv3_3 = resnet_block(ch3, kernel_size=ks)

        # This scales the fused heatmap
        # Opt. using conved merged heatmap instead
        self.fscale1_1 = conv(ch1, ch2, kernel_size=ks, stride=4)
        self.fscale1_2 = resnet_block(ch2, kernel_size=ks)
        self.fscale1_3 = resnet_block(ch2, kernel_size=ks)

        self.kconv_warp = KernelConv2D.KernelConv2D(kernel_size=ks_2d)
        self.kconv_deblur = KernelConv2D.KernelConv2D(kernel_size=ks_2d)

        # decoder
        self.upconv2_u = upconv(2*ch3, ch2)
        self.upconv2_2 = resnet_block(ch2, kernel_size=ks)
        self.upconv2_1 = resnet_block(ch2, kernel_size=ks)

        self.upconv1_u = upconv(ch2, ch1)
        self.upconv1_2 = resnet_block(ch1, kernel_size=ks)
        self.upconv1_1 = resnet_block(ch1, kernel_size=ks)

        self.img_prd = conv(ch1, 3, kernel_size=ks)

        #############################
        # Kernel Prediction Branch
        #############################

        # kernel network
        self.kconv1_1 = conv(12, ch1, kernel_size=ks, stride=1)
        # self.kconv1_1 = conv(9, ch1, kernel_size=ks, stride=1)
        self.kconv1_2 = resnet_block(ch1, kernel_size=ks)
        self.kconv1_3 = resnet_block(ch1, kernel_size=ks)

        self.kconv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.kconv2_2 = resnet_block(ch2, kernel_size=ks)
        self.kconv2_3 = resnet_block(ch2, kernel_size=ks)

        self.kconv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.kconv3_2 = resnet_block(ch3, kernel_size=ks)
        self.kconv3_3 = resnet_block(ch3, kernel_size=ks)

        self.fac_warp = nn.Sequential(
            conv(ch3, ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            conv(ch3, ch3 * ks_2d ** 2, kernel_size=1))

        self.kconv4 = conv(ch3 * ks_2d ** 2, ch3, kernel_size=1)

        self.fac_deblur = nn.Sequential(
            conv(2*ch3, ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            conv(ch3, ch3 * ks_2d ** 2, kernel_size=1))

        #############################
        # Interactive Landmark Branch
        #############################
        # self.HG = FeedbackHourGlass(256, 68)
        self.HG2 = KFSGNet(feature_num=256)
        self.fusion_block1 = FeatureHeatmapFusingBlock(feat_channel_in=ch1,
                                                       num_heatmap=5,
                                                       num_block=7)
        # self.fusion_block2 = FeatureHeatmapFusingBlock(feat_channel_in=ch3,
        #                                                num_heatmap=5,
        #                                                num_block=7)
        self.compress_in = ConvBlock(320, 256,
                                     kernel_size=1,
                                     act_type='prelu', norm_type=None)

        self.rescale = nn.Upsample(size=(64, 64), mode='bilinear')

        # self.aggregation_layer = LinearWeightedAvg(3)

    def forward(self, img_blur, last_img_blur, output_last_img, output_last_fea, fusion_last_feature):
        # For experiment in low-memory server
        # torch.cuda.empty_cache()
        out_img_list = []
        heatmap_list = []
        recovered = img_blur
        for i in range(self.num_steps):
            heatmap_input = self.HG2(recovered)
            heatmap_input_merge5 = merge_heatmap_5(heatmap_input, False)
            merge = torch.cat([img_blur, recovered, last_img_blur, output_last_img], 1)

            #############################
            # Feature Extraction
            #############################
            conv1_d = self.conv1_1(recovered)
            conv1_d = self.conv1_3(self.conv1_2(conv1_d))

            kconv1 = self.kconv1_3(self.kconv1_2(self.kconv1_1(merge)))
            kconv2 = self.kconv2_3(self.kconv2_2(self.kconv2_1(kconv1)))
            kconv3 = self.kconv3_3(self.kconv3_2(self.kconv3_1(kconv2)))
        
            # Kernel predicition
            kernel_warp = self.fac_warp(kconv3)
            kconv4 = self.kconv4(kernel_warp)
            kernel_deblur = self.fac_deblur(torch.cat([kconv3, kconv4],1))

            # Attentive fusion
            fused = self.fusion_block1(conv1_d, heatmap_input_merge5)
            conv1_d = fused

            # rescale feature
            feature = self.fscale1_3(self.fscale1_2(self.fscale1_1(fused)))

            # Deblur branch
            conv2_d = self.conv2_1(conv1_d)
            conv2_d = self.conv2_3(self.conv2_2(conv2_d))
            conv3_d = self.conv3_1(conv2_d)
            conv3_d = self.conv3_3(self.conv3_2(conv3_d))

            conv3_d_k = self.kconv_deblur(conv3_d, kernel_deblur)
        
            # Output_last fea comes from the previous frame, will only update at final
            if output_last_fea is None:
                output_last_fea = torch.cat([conv3_d, conv3_d],1)
        
            output_last_fea = self.fea1(output_last_fea)
            conv_a_k = self.kconv_warp(output_last_fea, kernel_warp)
 
            if fusion_last_feature is not None:
                conv3 = self.compress_in(torch.cat([conv3_d_k, conv_a_k, fusion_last_feature],1))
            else:
                conv3 = self.compress_in(torch.cat([conv3_d_k, conv_a_k, feature],1))

            # step-wise update
            output_last_fea = conv3
            fusion_last_feature = feature
            # decoder
            upconv2 = self.upconv2_1(self.upconv2_2(self.upconv2_u(conv3)))
            upconv1 = self.upconv1_1(self.upconv1_2(self.upconv1_u(upconv2)))
            output_img = self.img_prd(upconv1) + recovered

            # Update heatmap
            heatmap_input = self.HG2(output_img)
            heatmap_input_merge5 = merge_heatmap_5(heatmap_input, False)
            recovered = output_img
            out_img_list.append(output_img)
            heatmap_list.append(heatmap_input)
            
        
        # update frame-wise feature in the end
        output_fea = conv3
        fusion_feature = feature

        return out_img_list, output_fea, heatmap_list, fusion_feature