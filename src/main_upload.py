from skimage.feature import canny
from skimage.transform import resize
import sys
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import gradio as gr
import os
import matplotlib.pyplot as plt
from matplotlib.colors import *
from matplotlib.collections import LineCollection
from matplotlib.pyplot import get_cmap
from PIL import Image
from transformers import CLIPTextModel
from google.cloud import storage
import numpy as np
import torch


print(torch.cuda.is_available())
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
cache_dir = os.path.join(script_dir, 'models')

models_dir = os.path.join(script_dir, 'models')


# url = "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth"  # can also be a local path
# control_net_model = "control_v11p_sd15_canny.pth"
# control_net_model = os.path.join(models_dir, "control_v11p_sd15_canny.pth")
# controlnet = ControlNetModel.from_single_file(control_net_model,
#                                               cache_dir=cache_dir,
#                                               local_files_only=False,
#                                               safety_checker=None)
#
# clip_model = "openai/clip-vit-large-patch14"
# clip_transformer = CLIPTextModel.from_pretrained(clip_model, revision='0993c71e8ad62658387de2714a69f723ddfffacb')
# # url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.safetensors"  # can also be a local path
# # stablediff_model = "epicrealism_naturalSinRC1VAE.safetensors"
# stablediff_model = os.path.join(models_dir, "epicrealism_naturalSinRC1VAE.safetensors")
# pipe = StableDiffusionControlNetPipeline.from_single_file(
#     stablediff_model,
#     controlnet=controlnet,
#     cache_dir=cache_dir,
#     local_files_only=False,
#     text_encoder=clip_transformer)
# pipe.safety_checker = None


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )


def plot_helper(data_image, axs, drag_index, linewidths=3):
    """
    Plots stuff
    """
    x, y = data_image.real, data_image.imag
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    cmap = get_cmap('twilight')
    # norm = plt.Normalize(0.75, 0.8*len(x))
    norm = plt.Normalize(0.0*len(x), 1 * len(x))
    # norm = BoundaryNorm([coeff*len(x) for coeff in np.linspace(-0.2, 0.8, 20)],

    # data_image_grad = np.gradient(data_image)
    # norm = 100 * abs(data_image_grad)
    #                     ncolors=cmap.N//1.5)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=linewidths)

    # col = MplColorHelper('bone', 0, len(x))
    # colors = col.get_rgb(segments)
    # lc = LineCollection(segments, colors=colors, linewidths=linewidths)

    # Set the values used for colormapping
    lc.set_array(list(range(len(x))))
    lc.set_linewidth(2)
    line = axs.add_collection(lc)

    return axs


def test4(func,
          drag_min=0.80,
          drag_max=0.99,
          drag_num=500,
          t_min=2*np.pi,
          t_max=4*np.pi,
          t_steps=1000,
          total_terms=9,
          rad_sin_freq=1,
          rad_cos_freq=1,
          noise_num=0,
          radius_init=0,
          rad_sin_amp=0,
          rad_cos_amp=0,
          show_plot=False,
          border_amt=0.1):

    t = np.linspace(t_min, t_max, t_steps)

    xmax_scale = xmin_scale = ymin_scale = ymax_scale = border_amt

    fig = plt.figure(constrained_layout=False,
                     frameon=False,
                     figsize=(8, 8))
    plt.gca().set_aspect('equal')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.08, 0.08, 0.055))

    data_inputspace = (radius_init
                       + rad_sin_amp * np.sin(rad_sin_freq * t)
                       + rad_cos_amp * np.cos(rad_cos_freq * t)) * np.exp(1j * t ** 1.5)
    data_image = func(drag_max * data_inputspace, total_terms)

    x_len = np.abs(data_image.real.max() - data_image.real.min())
    y_len = np.abs(data_image.imag.max() - data_image.imag.min())

    img_bounds = [
        data_image[data_image.real.argmin()].real - xmin_scale * x_len,
        data_image[data_image.real.argmax()].real + xmax_scale * x_len,
        data_image[data_image.imag.argmin()].imag - ymin_scale * y_len,
        data_image[data_image.imag.argmax()].imag + ymax_scale * y_len]
    print(x_len, y_len, img_bounds)
    # ax = add_scatter_noise(ax, int(noise_num), img_bounds)

    for drag_index, drag in enumerate(np.linspace(drag_min, drag_max, drag_num)):
        data_image = func(drag * data_inputspace, total_terms)
        ax = plot_helper(data_image, ax, drag_index)

    plt.xlim([img_bounds[0],
              img_bounds[1]])
    plt.ylim([img_bounds[2],
              img_bounds[3]])

    # plt.cla()
    # for drag_index, drag in enumerate(np.linspace(drag_min, drag_max, drag_num)):
    #     data_image = drag * data_inputspace
    #     ax = plot_helper(data_image, ax, drag_index)
    image_path = os.path.join(script_dir, 'images', 'test_fig.png')
    plt.savefig(image_path, format='png')
    im = Image.open(image_path)
    return im


def g_better(z, N):
    """
    Returns a matrix with the sum of the power series operated on z with a max of N times
    """
    powers_to_use = np.geomspace(1, np.power(2, N), num=N + 1, dtype=int)
    coeffs = np.zeros(np.max(powers_to_use) + 1)
    coeffs[powers_to_use] = 1
    return np.polynomial.polynomial.polyval(z, coeffs)


def create_template(rad_cos_freq: float, drag_min: float, drag_max: float, drag_num: int, t_steps: int, border_amt: float):
    return test4(
        g_better,
        drag_min=drag_min,
        drag_max=drag_max,
        drag_num=drag_num,
        t_min=0.0 * np.pi,
        t_max=2.0 * np.pi,
        t_steps=t_steps,
        total_terms=9,
        rad_cos_freq=rad_cos_freq,
        rad_sin_freq=3,
        rad_sin_amp=1,
        rad_cos_amp=1,
        border_amt=border_amt,
        )


def run_stable_diff(prompt: str,
                    image,
                    num_inference_steps: int,
                    guidance_scale: float,
                    sigma: float,
                    low_threshold: float,
                    high_threshold: float):

    if high_threshold < low_threshold:
        high_threshold = np.random.uniform(low_threshold, 500)

    image_canny = canny(image[:, :, 0],
                        sigma=sigma,
                        low_threshold=low_threshold,
                        high_threshold=high_threshold,)
    image_canny_reshape = resize(image_canny, (512,
                                               512,
                                               3)
                                 ).astype(float)
    plt.imshow(image_canny_reshape)
    image_path = os.path.join(script_dir, 'images', 'test_fig.png')
    plt.savefig(image_path, format='png')
    return_image = pipe(prompt,
                        [image_canny_reshape],
                        height=512,
                        width=512,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        )[0][0]
    plt.close()
    plt.imshow(return_image)
    image_path = os.path.join(script_dir, 'images', 'return_fig.png')
    plt.savefig(image_path, format='png')
    return return_image


def upload_func(image):
    plt.imshow(image)
    image_path = os.path.join(script_dir, 'images', 'test_fig_saved.png')
    plt.savefig(image_path, format='png')

    upload_blob('public_controlnet_images',
                image_path,
                'test_img.png')


def main():
    """
    Runs one round of algart image creation
    """
    print("Creating Template")
    image = create_template(
        np.random.uniform(0, 10),
        np.random.uniform(0, 1.0),
        np.random.uniform(0, 0.99),
        np.random.randint(10, 100),
        np.random.randint(500, 10000),
        0.01)

    print("Running Stable Diffusion")
    # stable_diff_output = run_stable_diff(
    #     "A picture of an abstract glass art installation",
    #     np.array(image),
    #     15,
    #     np.random.uniform(0, 20),
    #     np.random.uniform(0, 20),
    #     np.random.uniform(0, 500),
    #     np.random.uniform(0, 500),
    # )

    print("Uploading Image")
    upload_func(np.array(image))

    print("Done!")

if __name__ == "__main__":
    main()
