import os

import torch
from PIL import Image

from utils.StyleEnc import StyleEncoder
from utils.image_datasets import center_crop_arr

import blobfile as bf

import numpy as np

from utils.nn import mean_flat


def load_model(sty_encoder_path):
    model = StyleEncoder(sty_dim=128)
    print("loading pre-trained style encoder...")
    checkpoint = torch.load(sty_encoder_path, map_location='cpu')
    tmp_dict = {}
    for k, v in checkpoint.items():
        if k in model.state_dict():
            tmp_dict[k] = v
    model.load_state_dict(tmp_dict)
    print(model)
    print("loading pre-trained style encoder successfully!")
    return model


def load_images(data_dir):
    # 传入数据集根目录
    print("loading learned sty image")
    style_font = os.listdir(data_dir)
    images = {}
    for sf in style_font:
        fonts = os.listdir(os.path.join(data_dir, sf))
        if len(fonts) != 0:
            sty_cls = sf.split('_')[1]
            image_path = os.path.join(data_dir, sf, fonts[0])
            single_image = load_single_image(image_path)
            if single_image is not None:
                images[int(sty_cls)] = single_image
    print("successfully load learned sty cls {}".format(images.keys()))
    return images


def load_single_image(path):
    image = None
    with bf.BlobFile(path, "rb") as i:
        image = Image.open(i)
        image.load()
        image.convert("RGB")
        sty_arr = center_crop_arr(image, 80)
        sty_arr = sty_arr.astype(np.float32) / 127.5 - 1
        sty_arr = np.transpose(sty_arr, [2, 0, 1])
        image = torch.from_numpy(sty_arr)
    return image


def find_similar_sty(unknow_sty_path, data_dir, encoder_ckpt_path):
    with torch.no_grad():
        encoder = load_model(encoder_ckpt_path).eval()
        images = load_images(data_dir)
        unknow_sty_image = load_single_image(unknow_sty_path)
        base_shape = unknow_sty_image.shape
        unknow_sty_feature = encoder(torch.reshape(unknow_sty_image, [1, base_shape[0], base_shape[1], base_shape[2]]))

        distance = float('inf')

        most_similar_cls = None

        for cls, gt in images.items():
            learn_sty_feature = encoder(torch.reshape(gt, [1, base_shape[0], base_shape[1], base_shape[2]]))
            d = mean_flat((unknow_sty_feature - learn_sty_feature) ** 2)
            print("similar with sty {} distance {}".format(cls, d))
            if d < distance:
                distance = d
                most_similar_cls = cls
        if most_similar_cls is not None:
            print("most similar sty cls {} path {}".format(most_similar_cls,
                                                           os.path.join(data_dir, 'id_{}'.format(most_similar_cls))))
        else:
            print("error")


if __name__ == '__main__':
    find_similar_sty('../result/00009.png', '../data_dir', '../pretrained_models/chinese_styenc.ckpt')