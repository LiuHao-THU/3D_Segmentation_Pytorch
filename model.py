import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_class = 2):
        super().__init__()
        # conv1
        self.conv1_1 = nn.Conv3d(1, 8, 3, padding=60)
        # torch.nn.Conv3d(in_channels, out_channels, kernel_size,
        # stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv3d(8, 8, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv3d(8, 16, 3, padding=15)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv3d(16, 16, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv3d(16, 32, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv3d(32, 32, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv3d(32, 32, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv3d(32, 64, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv3d(64, 64, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv3d(64, 512, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout3d()

        # fc7
        self.fc7 = nn.Conv3d(512, 512, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout3d()

        self.score_fr = nn.Conv3d(512, n_class, 1)
        self.score_pool3 = nn.Conv3d(32, n_class, 1)
        self.score_pool4 = nn.Conv3d(64, n_class, 1)

        self.upscore2 = nn.ConvTranspose3d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose3d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose3d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()


    def get_upsampling_weight(self, in_channels, out_channels, kernel_size):
        """Make a 2D bilinear kernel suitable for upsampling"""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size),
                          dtype=np.float64)
        weight[range(in_channels), range(out_channels), :, :, :] = filt
        return torch.from_numpy(weight).float()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight.data.zero_()
                m.weight.data.normal_(0.0, 0.1)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose3d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = self.get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             m.weight.data.zero_()
    #             m.weight.data.normal_(0.0, 0.1)
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #                 m.bias.data.normal_(0.0, 0.1)
    #         if isinstance(m, nn.ConvTranspose3d):
    #             m.weight.data.zero_()
    #             m.weight.data.normal_(0.0, 0.1)
    #             # assert m.kernel_size[0] == m.kernel_size[1]
    #             # initial_weight = self.get_upsampling_weight(
    #             #     m.in_channels, m.out_channels, m.kernel_size[0])
    #             # m.weight.data.copy_(initial_weight)
    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            # print("what is l1? ", l1)
            # print("what is l2? ", l2)
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))

    def forward(self, x):
        h = x
        # happyprint("init: ", x.data[0].shape)

        h = self.relu1_1(self.conv1_1(h))
        # happyprint("after conv1_1: ", h.data[0].shape)
        
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        # happyprint("after pool1: ", h.data[0].shape)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        # happyprint("after pool2: ", h.data[0].shape)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        # happyprint("after pool3: ", h.data[0].shape)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        # happyprint("after pool4: ", h.data[0].shape)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        # happyprint("after pool5: ", h.data[0].shape)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        # happyprint("after score_fr: ", h.data[0].shape)
        h = self.upscore2(h)

        # happyprint("after upscore2: ", h.data[0].shape)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
        # happyprint("after score_pool4: ", h.data[0].shape)

        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3], 5:5 + upscore2.size()[4]]

        score_pool4c = h  # 1/16
        # happyprint("after score_pool4c: ", h.data[0].shape)

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8
        # happyprint("after upscore_pool4: ", h.data[0].shape)

        h = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
        h = h[:, :,
             9:9 + upscore_pool4.size()[2],
             9:9 + upscore_pool4.size()[3],
             9:9 + upscore_pool4.size()[4]]
        score_pool3c = h  # 1/8
        # happyprint("after score_pool3: ", h.data[0].shape)

        # print(upscore_pool4.data[0].shape)
        # print(score_pool3c.data[0].shape)

        # Adjusting stride in self.upscore2 and self.upscore_pool4
        # and self.conv1_1 can change the tensor shape (size).
        # I don't know why!

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h) # dim: 88^3
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3], 31:31 + x.size()[4]].contiguous()
        # happyprint("after upscore8: ", h.data[0].shape)
        return h
