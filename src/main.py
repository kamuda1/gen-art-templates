from skimage.feature import canny
from skimage.transform import resize
import sys
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import os
import matplotlib.pyplot as plt
from matplotlib.colors import *
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from google.cloud import storage

from src.gcp_utils import upload_blob
from src.generative_template import create_template
from pathlib import Path


class MainApp:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.models_dir = os.path.join(self.script_dir, 'models')

        storage_client = storage.Client(os.getenv('PROJECT_NAME'))
        self.bucket = storage_client.get_bucket(os.getenv('MODEL_BUCKET'))

        self.control_net_filepath = os.path.join(self.models_dir, "control_sd15_canny.pth")
        self.stablediff_model_filepath = os.path.join(self.models_dir, "epicrealism_naturalSinRC1VAE.safetensors")
        self.clip_model_path = os.path.join(self.models_dir, "openai", "clip-vit-large-patch14")
        self.download_models()
        self.initialize_models()

    def download_files_from_bucket(self, bucket, bucket_path):
        pass

    def download_models(self):

        if not os.path.isfile(self.control_net_filepath):
            canny_blob = self.bucket.blob("models/control_sd15_canny.pth")
            canny_blob.download_to_filename(self.control_net_filepath)

        if not Path(self.clip_model_path).is_dir():
            self.download_files_from_bucket(self.bucket, "openai/clip-vit-large-patch14")

        if not os.path.isfile(self.stablediff_model_filepath):
            stablediffusion_blob = self.bucket.blob("models/epicrealism_naturalSinRC1VAE.safetensors")
            stablediffusion_blob.download_to_filename(self.stablediff_model_filepath)

    def initialize_models(self):
        controlnet = ControlNetModel.from_single_file(
            self.control_net_filepath,
            cache_dir=self.models_dir,
            local_files_only=True,
            safety_checker=None)

        clip_transformer = CLIPTextModel.from_pretrained(self.clip_model_path, local_files_only=True)
        clip_tokenizer = CLIPTokenizer.from_pretrained(self.clip_model_path, local_files_only=True)

        # scheduler_type ["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm", "ddim"]
        self.sd_pipe = StableDiffusionControlNetPipeline.from_single_file(
            self.stablediff_model_filepath,
            controlnet=controlnet,
            cache_dir=self.models_dir,
            local_files_only=True,
            text_encoder=clip_transformer,
            tokenizer=clip_tokenizer,
            scheduler_type='ddim',
            load_safety_checker=False)

        self.sd_pipe.safety_checker = None

    def run_stable_diff(self, prompt, image, num_inference_steps, guidance_scale, sigma, low_threshold, high_threshold):

        image_canny = canny(image[:, :, 0],
                            sigma=sigma,
                            low_threshold=low_threshold,
                            high_threshold=high_threshold,)
        image_canny_reshape = resize(image_canny, (512,
                                                   512,
                                                   3)
                                     ).astype(float)
        plt.imshow(image_canny_reshape)
        image_path = os.path.join(self.script_dir, 'images', 'test_fig.png')
        plt.savefig(image_path, format='png')
        return_image = self.sd_pipe(prompt,
                                    [image_canny_reshape],
                                    height=512,
                                    width=512,
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale,
                                    )[0][0]
        plt.close()
        plt.imshow(return_image)
        image_path = os.path.join(self.script_dir, 'images', 'return_fig.png')
        plt.savefig(image_path, format='png')
        return return_image

    def run(self):
        im_template = create_template(5, 0.3, 0.4, 20, 500, 0.01, self.script_dir)
        template_filepath = os.path.join(self.script_dir, 'images', 'template_image.png')

        im_template = np.asarray(Image.open(template_filepath))

        prompt = "Modern art, blue"
        self.run_stable_diff(
            prompt,
            im_template,
            num_inference_steps=5,
            guidance_scale=1.0,
            sigma=2.0,
            low_threshold=0.0,
            high_threshold=16.0)

        output_image_path = os.path.join(self.script_dir, 'images', 'return_fig.png')
        upload_blob('public_controlnet_images',
                    output_image_path,
                    'test_img1.png')


if __name__ == "__main__":
    main_app = MainApp()
    main_app.run()
