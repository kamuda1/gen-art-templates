import os
from matplotlib.colors import *
from skimage.feature import canny
from skimage.transform import resize
import gradio as gr
# from utils import *
import sys
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel


script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
cache_dir = os.path.join(script_dir, 'cache_dir')

models_dir = os.path.join(script_dir, 'models')


# url = "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth"  # can also be a local path
control_net_model = os.path.join(models_dir, "control_v11p_sd15_canny.pth")
controlnet = ControlNetModel.from_single_file(control_net_model,
                                              cache_dir=cache_dir,
                                              local_files_only=False,
                                              safety_checker=None)


# url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.safetensors"  # can also be a local path
stablediff_model = os.path.join(models_dir, "epicrealism_naturalSinRC1VAE.safetensors")
pipe = StableDiffusionControlNetPipeline.from_single_file(
    stablediff_model,
    controlnet=controlnet,
    cache_dir=cache_dir,
    local_files_only=False)
pipe.safety_checker = None

def create_template(rad_cos_freq, drag_min, drag_max, drag_num, t_steps, border_amt):
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


def run_stable_diff(prompt, image, num_inference_steps):

    image_canny = canny(image[:, :, 0])
    image_canny_reshape = resize(image_canny, (image_canny.shape[0],
                                               image_canny.shape[0],
                                               3)
                                 ).astype(float)

    return_image = pipe(prompt,
                        [image_canny_reshape],
                        height=256,
                        width=256,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=7.5,
                        )[0][0]

    return return_image


with gr.Blocks() as demo:
    gr.Markdown("Create Stable Diffusion images with AlgoArt.")
    with gr.Tab("Generate AlgoArt") as algoarttab:
        algoart_row = gr.Interface(
            fn=create_template,
            inputs=[gr.Slider(0, 10, step=0.1, value=5),
                    gr.Slider(0, 1.0, step=0.1, value=0.3),
                    gr.Slider(0, 0.99, step=0.05, value=0.4),
                    gr.Slider(10, 100, step=5, value=20),
                    gr.Slider(500, 10000, step=500, value=500),
                    gr.Dropdown([0.01, 0.1], value=0.01),
                    ],
            outputs="image")
    with gr.Tab("Stable Diffusion"):
        gr.Interface(
            fn=run_stable_diff,
            inputs=['text',
                    algoart_row.output_components[0],
                    gr.Slider(5, 50, step=5, value=15)
                    ],
            outputs="image")


if __name__ == "__main__":
    demo.launch(show_api=False, server_name="0.0.0.0", server_port=8000)



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
    image_path = os.path.join('images', 'test_fig.png')
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
