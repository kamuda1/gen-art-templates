import logging
import random

import torch
from diffusers import StableDiffusionPipeline
from config import config

logger = logging.getLogger(__name__)


class Diffuser:  # pylint: disable=too-few-public-methods
    """
    A container to encapsulate

    Args:
        model_name (str): The huggingface model name to use for image generation.
    """

    config_settings: dict

    def __init__(self, model_name: str = "stabilityai/stable-diffusion-2-1") -> None:
        logger.info("CUDA is available: %s", torch.cuda.is_available())
        self.config_settings = config["diffuser"]["settings"]
        self.cuda_is_available = torch.cuda.is_available()

        if not self.config_settings["use_gpu"]:
            self.pipeline = StableDiffusionPipeline.from_pretrained(model_name)
            return

        if self.config_settings["use_half_precision"]:
            logger.info("Using half-precision 16-bit floats.")
            torch_dtype = torch.float16  # pylint: disable=no-member
        else:
            torch_dtype = torch.float32  # pylint: disable=no-member

        self.pipeline = StableDiffusionPipeline.from_pretrained(model_name,
                                                                torch_dtype=torch_dtype)
        self.pipeline.to("cuda")

        if self.config_settings["enable_xformers_attention"]:
            logger.info("Enabling Xformers memory efficient attention.")
            self.pipeline.enable_xformers_memory_efficient_attention()

    def make_image(self, prompt: str, cfg: float, steps: int) -> str:
        """Call the huggingface pipeline with several arguments and save the resulting image to disk as "img.png"

        Args:
            prompt (str): The prompt string.
            cfg (float): The float indicating the strength for "context free guidance".
            steps (int): The number of diffusion steps to perform.

        Return:
            str: The file_name string
        """

        image = self.pipeline(prompt=prompt, guidance_scale=cfg, num_inference_steps=steps).images[0]
        file_name = f"output_images/img_{random.randint(1, 10**10)}.png"
        image.save(file_name)
        return file_name