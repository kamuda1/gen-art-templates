import gradio as gr
from utils import *
import sys
# from diffuser import Diffuser

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

    return_image = pipe(prompt,
                        [image],
                        height=256,
                        width=256,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=7.5,
                        )[0][0]

    print(type(return_image))
    print(return_image)

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
