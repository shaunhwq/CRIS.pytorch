# Note: This is not optimized, if you want to do multiple inference should instantiate the model once only.
import os
import argparse
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

from model import CRISModified
from utils.dataset import tokenize


def rm_module_prefix(state_dict: dict):
    """
    Remove module prefix from pytorch weights
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove the 'module.' prefix
        new_state_dict[new_key] = value
    return new_state_dict


def get_transform_mat(input_size, img_size, inverse=False):
    ori_h, ori_w = img_size
    inp_h, inp_w = input_size
    scale = min(inp_h / ori_h, inp_w / ori_w)
    new_h, new_w = ori_h * scale, ori_w * scale
    bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

    src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
    dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                    [bias_x, new_h + bias_y]], np.float32)

    mat = cv2.getAffineTransform(src, dst)
    if inverse:
        mat_inv = cv2.getAffineTransform(dst, src)
        return mat, mat_inv
    return mat, None


def pre_process(
    image: np.array,
    sentence: str,
    word_len: int = 17,
    device: str = "cuda:0",
    mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
    std: List[float] = [0.26862954, 0.26130258, 0.27577711],
    input_size: List[float] = [416, 416],
):
    # Pre-process image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to model's input size
    mat, mat_inv = get_transform_mat(input_size, image.shape[:2], True)
    image = cv2.warpAffine(image, mat, input_size, flags=cv2.INTER_CUBIC, borderValue=[val * 255 for val in mean])

    image = torch.from_numpy(np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).float().unsqueeze(0)

    mean = torch.tensor(mean).reshape(3, 1, 1)
    std = torch.tensor(std).reshape(3, 1, 1)
    image.div_(255.).sub_(mean).div_(std)

    # Pre process text
    word_vec = tokenize(sentence, word_len, True).squeeze(0)
    word_vec = word_vec.unsqueeze(0)

    image = image.to(device)
    word_vec = word_vec.to(device)

    return image, word_vec


def post_process(model_pred: torch.tensor):
    model_pred = model_pred.cpu().squeeze().numpy()
    model_pred = (model_pred * 255.0).clip(0, 255).astype(np.uint8)
    return model_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # File related
    parser.add_argument("-ii", "--input_image", type=str, required=True, help="Path to input image containing images")
    parser.add_argument("-ip", "--input_prompt", type=str, required=True, help="Prompt accompanying the image")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path to output folder")
    # Model related
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Device to use e.g. 'cuda:0', 'cuda:1', 'cpu'")
    parser.add_argument("-w", "--weights", type=str, default="pretrain/cris_r50.pt", help="Path to CRIS pretrained weights")
    parser.add_argument("-cw", "--clip_weights", type=str, default="pretrain/RN50.pt", help="Path to CLIP pretrained weights")

    parser.add_argument("--word_length", type=int, default=17, help="Max length of word")

    args = parser.parse_args()

    # Prepare output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare model. Using r50 by default (set through init params)
    model = CRISModified(clip_weights_path=args.clip_weights)
    model_weights = torch.load(args.weights)["state_dict"]
    model_weights = rm_module_prefix(model_weights)
    model.load_state_dict(model_weights)
    model.eval()
    model.to(args.device)

    # Model inference
    with torch.no_grad():
        in_image = cv2.imread(args.input_image, cv2.IMREAD_UNCHANGED)

        model_inputs = pre_process(in_image, args.input_prompt, args.word_length, args.device)

        # Inference
        pred = model(*model_inputs)
        pred = torch.sigmoid(pred)
        pred = F.interpolate(pred, size=[416, 416], mode='bicubic', align_corners=True).squeeze(1)

        segmentation_image = post_process(pred)

        basename = os.path.basename(args.input_image)
        name, ext = os.path.splitext(basename)
        output_path = os.path.join(args.output_dir, name + ".png")
        cv2.imwrite(output_path, segmentation_image)
