import torch.nn as nn
def calc_mean_std(features):
    """
    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 0, 0]
    """
    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std


def adain(content_features,style_mean, style_std):
    """
    Adaptive Instance Normalization

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features

# adain的模块化，用于嵌入UNET中
class adain_module(nn.Module):
    def __init__(self):
        super(adain_module, self).__init__()
    def forward(self,x,style_mean,style_std):
        return adain(x,style_mean,style_std)