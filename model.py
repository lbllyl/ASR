import torch
import torch.nn as nn
import torchvision.models as models


class ASR(nn.Module):
    def __init__(self):
        super(ASR, self).__init__()
        self.feature_extractor_1 = FeatureExtractor()
        self.feature_extractor_2 = FeatureExtractor()
        self.feature_extractor_3 = FeatureExtractor()
        self.feature_extractor_4 = FeatureExtractor()
        self.fc1 = Classifier()
        self.fc2 = Classifier()
        self.fc3 = Classifier()
        self.fc4 = Classifier()
        self.feature_fuser = FeatureFuser(in_channels=1536, out_channels=512, output_size=7)
        self.conv1x1 = nn.Conv2d(512 + 1024 + 1408, 512, kernel_size=1)

    def forward(self, image_groups):
        feature_1_list = []
        feature_2_list = []
        feature_3_list = []

        for group_idx in range(4):
            image_batch = image_groups[:, group_idx, :, :, :, :]
            image_batch = image_batch.view(-1, 3, 3, 224, 224)  # 调整图像批次的形状
            
            feature_extractor = getattr(self, f'feature_extractor_{group_idx + 1}')

            # 输入feature_extractor_{group_idx + 1}.resnet18
            feature_1_list.append(feature_extractor.resnet18(image_batch[:, 0, :, :, :]))
            feature_2_list.append(feature_extractor.densenet121(image_batch[:, 1, :, :, :]))
            feature_3_list.append(feature_extractor.efficientnetb2(image_batch[:, 2, :, :, :]))

        feature_1 = feature_max(feature_1_list)
        feature_2 = feature_max(feature_2_list)
        feature_3 = feature_max(feature_3_list)

        aap_out_1 = aap(feature_1)
        aap_out_2 = aap(feature_2)
        aap_out_3 = aap(feature_3)

        logits_1 = self.fc1(aap_out_1.view(-1, 512))
        logits_2 = self.fc2(aap_out_2.view(-1, 512))
        logits_3 = self.fc3(aap_out_3.view(-1, 512))

        # feature fuser
        feature_all = torch.cat((feature_1, feature_2, feature_3), dim=1)

        feature_fused = self.conv1x1(feature_all.view(feature_all.size(0), -1, 1, 1))

        aap_out_4 = aap(feature_fused)

        logits_4 = self.fc4(aap_out_4.view(-1, 512))

        return logits_1, logits_2, logits_3, logits_4


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.densenet121 = models.densenet121(pretrained=True)
        self.efficientnetb2 = models.efficientnet_b2(pretrained=True)
    
        self.resnet18.fc = nn.Identity()
        self.densenet121.classifier = nn.Identity()
        self.efficientnetb2.classifier = nn.Identity()

    def forward(self, x):
        f11 = self.resnet18(x)
        f21 = self.densenet121(x)
        f34 = self.efficientnetb2(x)
        return f11, f21, f34

class FeatureFuser(nn.Module):
    def __init__(self, in_channels, out_channels, output_size):
        super(FeatureFuser, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.aap = nn.AdaptiveAvgPool2d(output_size)
    
    def forward(self, f11, f12, f13, f14):
        f = torch.max(f11, f12)
        f = torch.max(f, f13)
        f = torch.max(f, f14)
        
        return f

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(512, 5)
    
    def forward(self, x):
        logits = self.fc(x)
        logits = torch.sigmoid(logits)
        return logits
    
def aap(fused):
    # 重塑特征张量为 4D
    fused = fused.view(fused.size(0), -1, 1, 1)
    
    aap = nn.AdaptiveAvgPool2d((1, 1))
    aap_out = aap(fused)
    
    # 重塑输出张量为 2D
    aap_out = aap_out.view(aap_out.size(0), -1)
    
    return aap_out

def feature_max(feature_list):
    f = torch.max(feature_list[0], feature_list[1])
    f = torch.max(f, feature_list[2])
    f = torch.max(f, feature_list[3])
    
    return f