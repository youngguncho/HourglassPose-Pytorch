import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


def model_parser(model, sum_mode=False, dropout_rate=0.0, bayesian=False):
    base_model = None

    if model == 'Resnet':
        base_model = models.resnet34(pretrained=True)
        network = HourglassNet(base_model, sum_mode, dropout_rate, bayesian)
    else:
        assert 'Unvalid Model'

    return network


class HourglassNet(nn.Module):
    def __init__(self, base_model, sum_mode=False, dropout_rate=0.0, bayesian=False):
        super(HourglassNet, self).__init__()

        self.bayesian = bayesian
        self.dropout_rate = dropout_rate
        self.sum_mode = sum_mode

        # Encoding Blocks
        self.init_block = nn.Sequential(*list(base_model.children())[:4])
        # self.res1_block = nn.Sequential(*list(base_model.layer1.children()))

        self.res_block1 = base_model.layer1
        self.res_block2 = base_model.layer2
        self.res_block3 = base_model.layer3
        self.res_block4 = base_model.layer4

        # Decoding Blocks
        if sum_mode:
            self.deconv_block1 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block2 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block3 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.conv_block = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        else:
            self.deconv_block1 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block2 = nn.ConvTranspose2d(512, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block3 = nn.ConvTranspose2d(256, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.conv_block = nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        # Regressor
        self.fc_dim_reduce = nn.Linear(56 * 56 * 32, 1024)
        self.fc_trans = nn.Linear(1024, 3)
        self.fc_rot = nn.Linear(1024, 4)

        # Initialize Weights
        init_modules = [self.deconv_block1, self.deconv_block2, self.deconv_block3, self.conv_block,
                        self.fc_dim_reduce, self.fc_trans, self.fc_rot]

        for module in init_modules:
            if isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # nn.init.normal_(self.fc_last.weight, 0, 0.01)
        # nn.init.constant_(self.fc_last.bias, 0)
        #
        # nn.init.normal_(self.fc_position.weight, 0, 0.5)
        # nn.init.constant_(self.fc_position.bias, 0)
        #
        # nn.init.normal_(self.fc_rotation.weight, 0, 0.01)
        # nn.init.constant_(self.fc_rotation.bias, 0)

    def forward(self, x):
        # Conv
        x = self.init_block(x)
        x_res1 = self.res_block1(x)
        x_res2 = self.res_block2(x_res1)
        x_res3 = self.res_block3(x_res2)
        x_res4 = self.res_block4(x_res3)

        # Deconv
        x_deconv1 = self.deconv_block1(x_res4)
        if self.sum_mode:
            x_deconv1 = x_res3 + x_deconv1
        else:
            x_deconv1 = torch.cat((x_res3, x_deconv1), dim=1)

        x_deconv2 = self.deconv_block2(x_deconv1)
        if self.sum_mode:
            x_deconv2 = x_res2 + x_deconv2
        else:
            x_deconv2 = torch.cat((x_res2, x_deconv2), dim=1)

        x_deconv3 = self.deconv_block3(x_deconv2)
        if self.sum_mode:
            x_deconv3 = x_res1 + x_deconv3
        else:
            x_deconv3 = torch.cat((x_res1, x_deconv3), dim=1)

        x_conv = self.conv_block(x_deconv3)
        x_linear = x_conv.view(x_conv.size(0), -1)
        x_linear = self.fc_dim_reduce(x_linear)
        x_linear = F.relu(x_linear)

        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            x_linear = F.dropout(x_linear, p=self.dropout_rate, training=dropout_on)

        trans = self.fc_trans(x_linear)
        rot = self.fc_rot(x_linear)

        return trans, rot
