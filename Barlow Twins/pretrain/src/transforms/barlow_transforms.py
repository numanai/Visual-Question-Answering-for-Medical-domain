from torchvision import transforms
import numpy as np
import torch

from PIL import Image, ImageOps, ImageFilter
import random

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class HistogramNormalize(object):
    """
    Apply histogram normalization.
    Args:
        number_bins: Number of bins to use in histogram.
    """

    def __init__(self, number_bins: int = 256):
        self.number_bins = number_bins

    def __call__(self, sample):
        image = sample.numpy()

        # get image histogram
        image_histogram, bins = np.histogram(
            image.flatten(), self.number_bins, density=True
        )
        cdf = image_histogram.cumsum()  # cumulative distribution function
        cdf = 255 * cdf / cdf[-1]  # normalize

        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
        image_equalized.reshape(image.shape)

        sample = torch.tensor(image_equalized.reshape(image.shape)).to(
            sample
        )

        return sample


class TensorToRGB(object):
    """
    Convert Tensor to RGB by replicating channels.
    Args:
        num_output_channels: Number of output channels (3 for RGB).
    """

    def __init__(self, num_output_channels: int = 3):
        self.num_output_channels = num_output_channels

    def __call__(self, sample):
        expands = list()
        for i in range(sample.ndim):
            if i == 0:
                expands.append(self.num_output_channels)
            else:
                expands.append(-1)
        sample = sample.expand(*expands)

        return sample

class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),

            #Color jittering is not appropriate for medical images
            transforms.RandomApply(
                 [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                         saturation=0.2, hue=0.1)],
                 p=0.8
             ),

            #Random Gray scale and Gaussian blur are not appropriate for medical images.
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            HistogramNormalize(),
            TensorToRGB(),

            # We will use normalization values based on imagenet because we are using models with weights pretrained with imagenet.
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225]),
        ])
        self.transform_prime = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),


            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            HistogramNormalize(),
            TensorToRGB(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2