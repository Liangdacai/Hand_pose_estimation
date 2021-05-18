import torch
from torch import nn

from hand_pose_model.conv import conv, conv_dw, conv_dw_no_bn


class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class BlazeBlock(nn.Module):
    def __init__(self, block_num = 3, in_channel=24,out_channel = 48, channel_padding = 1):
        super(BlazeBlock, self).__init__()

        # <----- downsample ----->
        self.downsample_a = conv_dw(in_channel, out_channel, stride=2)
        if channel_padding:
            self.downsample_b = nn.Sequential(
                nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
                conv_dw(in_channel,in_channel),
                conv_dw(in_channel,out_channel)
            )
        else:
            self.downsample_b = nn.MaxPool2d((3, 3), stride=(2, 2),padding=(1,1))

        self.conv = nn.ModuleList()
        for i in range(block_num):
            self.conv.append(conv_dw(out_channel, out_channel))

    def forward(self, x):
        x = self.downsample_a(x) + self.downsample_b(x)
        for i in range(len(self.conv)):
            x = x + self.conv[i](x)
        return x

class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, use_heatmap,num_refinement_stages=1, num_channels=128, num_heatmaps=22, num_pafs=48):
        super().__init__()

        self.use_heatmap = use_heatmap
        self.model = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )
        self.cpm = Cpm(512, num_channels)

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))
        #--------------------joints------------------#
        if not self.use_heatmap:
            self.joints1 = BlazeBlock(block_num=4, in_channel=22, out_channel=96)
            self.joints2 = BlazeBlock(block_num=5, in_channel=96, out_channel=192)  # input res: 32
            self.joints3 = BlazeBlock(block_num=6, in_channel=192, out_channel=288)  # input res: 16
            self.joints4 = nn.Sequential(
                BlazeBlock(block_num=7, in_channel=288, out_channel=288, channel_padding=0),
                BlazeBlock(block_num=7, in_channel=288, out_channel=288, channel_padding=0)
            )
            self.fc = nn.Linear(288, 2 * 21)


    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        if self.use_heatmap:
            stages_output = torch.nn.functional.interpolate(stages_output[-2], scale_factor=2, mode='bilinear',
                                                            align_corners=None)
            return stages_output
        else:
            x = self.joints1(stages_output[-2])
            x = self.joints2(x)
            x = self.joints3(x)
            x = self.joints4(x)
            joints = self.fc(x.reshape((-1, 288)))
            return joints






if __name__ == '__main__':
    net =  PoseEstimationWithMobileNet(3)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    input_size=(1, 3, 256, 256)
    # pip install --upgrade git+https://github.com/kuan-wang/pytorch-OpCounter.git
    from thop import profile

    x = torch.randn(input_size)
    flops, params = profile(net, [x])
    print(flops)
    print(params)
    print('Total params: %.2fM' % (params/1000000.0))
    print('Total flops: %.2fM' % (flops/1000000.0))
    x = torch.randn(input_size)
    import time

    for i in range(100):
        t1 = time.time()
        out = net(x)
        t2 = time.time()
        print(t2 - t1)


