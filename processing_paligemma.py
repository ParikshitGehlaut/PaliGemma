from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch 
from torch.nn import nn
from torch.nn.functional import F 

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def resize(
    image: Image,
    size: tuple(int, int),
    resample: Image.Resampling = None,
)-> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (height, width), resample=resample
    )
    return resized_image

def rescale(
    image: np.ndarray,
    scale: float = None, 
    dtype: np.dtype = np.float32
) -> np.ndarray:
    image = image * scale
    image = image.astype(dtype)
    return image

def normalize(
    image: np.ndarray,
    mean: union[float, Iterable[float]],
    std: union[float, Iterable[float]],
) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean)/std
    return image

def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[union[float, List[float]]] = None, 
    image_std: Optional[union[float, List[float]]] = None, 
) -> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]

    # convert each image into numpy array
    images = [np.array(image) for image in images]

    # rescale the pixel values to be in the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]

    # Normalize the images to have mean 0 and standard deviation 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]

    # move the channel dimension to the first dimension
    # model expects the image to have mean 0 and std 1
    images = [image.transpose(2, 0, 1) for image in images]
    return images

class PaliGemmaProcessor:

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ] # object detection
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ] # object segmentation

        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self, 
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = true,
    ) -> dict:
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor= 1/255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        ) # return the list of numpy arrays

        # convert the list of numpy arrays to a single numpy arrays with shape [Batch_size, channel, Height, width]
        pixel_values = np.stack(pixel_values, axis=0)
        # convert numpy array to a PyTorch tensor
        pixel_values = torch.tensor(pixel_values)

        # prepend a 'self.image_seq_length' number of image tokens to the prompt

        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt = prompt,
                bos_token = self.tokenizer.bos_token,
                image_seq_len = self.image_seq_length,
                image_token = self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_date = {"pixel_values": pixel_values, **inputs}

        return return_date