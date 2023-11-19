import gradio as gr
from utils import *


def create_template(rad_cos_freq, drag_min, drag_max, drag_num):
    return test4(
        g_better,
        drag_min=drag_min,
        drag_max=drag_max,
        drag_num=drag_num,
        t_min=0.0 * np.pi,
        t_max=2.0 * np.pi,
        t_steps=5000,
        total_terms=9,
        rad_cos_freq=rad_cos_freq,
        rad_sin_freq=3,
        rad_sin_amp=1,
        rad_cos_amp=1,
        )


def run_stable_diff(prompt, image):
    return image


with gr.Blocks() as demo:
    gr.Markdown("Create Stable Diffusion images with AlgoArt.")
    with gr.Tab("Generate AlgoArt") as algoarttab:
        algoart_row = gr.Interface(
            fn=create_template,
            inputs=[gr.Slider(0, 10, step=0.1, value=5),
                    gr.Slider(0, 1.0, step=0.1, value=0.8),
                    gr.Slider(0, 0.99, step=0.05, value=0.9),
                    gr.Slider(10, 100, step=5, value=60)
                    ],
            outputs="image")
    with gr.Tab("Stable Diffusion"):
        gr.Interface(
            fn=run_stable_diff,
            inputs=['text', algoart_row.output_components[0]],
            outputs="image")


if __name__ == "__main__":
    demo.launch(show_api=False, server_name="0.0.0.0", server_port=8000)
