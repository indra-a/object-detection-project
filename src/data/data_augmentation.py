import torch
import torchvision.transforms.functional as F
import numpy as np
from typing import Any

class Resize:
    """ Resizes input image to a 224 by 224 size"""
    def __init__(self, new_size = (224,224)):
        self.new_size = new_size

    def __call__(self, sample):
        w, h = sample[0].size
        image_new = F.resize(sample[0], (self.new_size[0], self.new_size[1]))
        label_1 = sample[1][0] * self.new_size[0] / w
        label_2 = sample[1][1] * self.new_size[1] / w
        return image_new, (label_1, label_2)
    

class RandomHorizontalFlip:
    """ Randomly performs horizontal flip on sample image """
    def __init__(self):
        pass

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        label = image_label_sample[1]
        w, h = image.size
        c_x, c_y = label
        if np.random.random() < 0.5:
            image = F.hflip(image)
            label = w - c_x, c_y
        return image, label
    
class RandomVerticalFlip:
    """ Randomly performs vertical flip on sample image """
    def __init__(self):
        pass

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        label = image_label_sample[1]
        w, h = image.size
        c_x, c_y = label
        if np.random.random() < 0.5:
            image = F.vflip(image)
            label = c_x, h - c_y
        return image, label
    
class RandomTranslation:
    """ Performs a random x, y translation on sample image """
    def __init__(self, max_translation=(0.2, 0.2)):
        if (not 0 <= max_translation[0] <= 1) or (not 0 <= max_translation[1] <= 1):
            raise ValueError(f'Variable max_translation should be float between 0 to 1')
        self.max_translation_x = max_translation[0]
        self.max_translation_y = max_translation[1]

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        label = image_label_sample[1]
        w, h = image.size
        c_x, c_y = label
        x_translate = int(np.random.uniform(-self.max_translation_x, self.max_translation_x) * w)
        y_translate = int(np.random.uniform(-self.max_translation_y, self.max_translation_y) * h)
        image = F.affine(image, translate=(x_translate, y_translate), angle=0, scale=1, shear=0)
        label = c_x + x_translate, c_y + y_translate
        return image, label
  
class ImageAdjustment:
    """ Randomly applies different level of brightness, contrast and gamma_factor on sample image"""
    def __init__(self, p=0.5, brightness_factor=0.8, contrast_factor=0.8, gamma_factor=0.4):
        if not 0 <= p <= 1:
            raise ValueError(f'Variable p is a probability, should be float between 0 to 1')
        self.p = p
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.gamma_factor = gamma_factor

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        label = image_label_sample[1]

        if np.random.random() < self.p:
            brightness_factor = 1 + np.random.uniform(-self.brightness_factor, self.brightness_factor)
            image = F.adjust_brightness(image, brightness_factor)

        if np.random.random() < self.p:
            contrast_factor = 1 + np.random.uniform(-self.brightness_factor, self.brightness_factor)
            image = F.adjust_contrast(image, contrast_factor)

        if np.random.random() < self.p:
            gamma_factor = 1 + np.random.uniform(-self.brightness_factor, self.brightness_factor)
            image = F.adjust_gamma(image, gamma_factor)

        return image, label
    

class ToTensor:
    """ Converts an Image to Tensor """
    def __init__(self, scale_label = True):
        self.scale_label = scale_label

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        label = image_label_sample[1]
        w, h = image.size
        c_x, c_y = label

        image = F.to_tensor(image)

        if self.scale_label:
            label = c_x/w, c_y/h
        label = torch.tensor(label, dtype = torch.float32)
        return image, label
    
class ToPILImage:
    """ Converts a Tensor to PIL Image """
    def __init__(self, unscale_label=True):
        self.unscale_label = unscale_label

    def __call__(self, image_label_sample):
        image = image_label_sample[0]
        label = image_label_sample[1].tolist()

        image = F.to_pil_image(image)
        w, h = image.size

        if self.unscale_label:
            c_x, c_y = label
            label = c_x*w, c_y*h

        return image, label