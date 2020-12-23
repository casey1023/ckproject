import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size, 0.8))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout = 0.0):
        super(UNetUp, self).__init__()

        layers = [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size, 0.8),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
        

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x



##############################
#           RESNET
##############################

class ResidualBlock(nn.Module):
    def __init__(self, in_features, dropout = 0.0):
        super(ResidualBlock, self).__init__()

        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


##############################
#        Generator
##############################
class Generator(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(Generator, self).__init__()

        channels, self.h, self.w = input_shape

        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256, dropout=0.3)
        self.down4 = UNetDown(256, 256, normalize=False, dropout=0.3)

        model = []
        numberlist = [3, 64, 128, 256, 256, 512, 256, 128]
        # Residual blocks
        for i in range(len(numberlist)):
            temp = []
            for j in range(num_residual_blocks):
                if j == 0 or j == num_residual_blocks - 1:
                    temp += [ResidualBlock(numberlist[i])]
                else:
                    temp += [ResidualBlock(numberlist[i], dropout=0.3)]
            model.append(temp)

        self.up1 = UNetUp(256, 256, dropout=0.3)
        self.up2 = UNetUp(512, 128, dropout=0.3)
        self.up3 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(128, channels, 3, stride=1, padding=1), nn.Tanh()
        )

        self.model0 = nn.Sequential(*(model[0]))
        self.model1 = nn.Sequential(*model[1])
        self.model2 = nn.Sequential(*model[2])
        self.model3 = nn.Sequential(*model[3])
        self.model4 = nn.Sequential(*model[4])
        self.model5 = nn.Sequential(*model[5])
        self.model6 = nn.Sequential(*model[6])
        self.model7 = nn.Sequential(*model[7])

    def forward(self, x):
        x = self.model0(x)
        d1 = self.down1(x)

        d1 = self.model1(d1)
        d2 = self.down2(d1)

        d2 = self.model2(d2)
        d3 = self.down3(d2)

        d3 = self.model3(d3)
        d4 = self.down4(d3)

        d4 = self.model4(d4)
        u1 = self.up1(d4, d3)

        u1 = self.model5(u1)
        u2 = self.up2(u1, d2)

        u2 = self.model6(u2)
        u3 = self.up3(u2, d1)

        u3 = self.model7(u3)
        return self.final(u3)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True, dropout = 0.0):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                layers.append(nn.Dropout(dropout))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128, dropout=0.3),
            *discriminator_block(128, 256, dropout=0.3),
            *discriminator_block(256, 512, dropout=0.3),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
