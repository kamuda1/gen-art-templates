import gradio as gr
import os
import matplotlib.pyplot as plt
from matplotlib.colors import *
import matplotlib as mpl
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.pyplot import get_cmap
from PIL import Image
from skimage.feature import canny
from skimage.transform import resize

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


class MplColorHelper:

  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)


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


def run_stable_diff(prompt, image):
    image_canny = canny(image[:, :, 0])
    image_canny_reshape = resize(image_canny, (image_canny.shape[0],
                                               image_canny.shape[0],
                                               3)
                                 )

    return image_canny.astype(float)


with gr.Blocks() as demo:
    gr.Markdown("Create Stable Diffusion images with AlgoArt.")
    with gr.Tab("Generate AlgoArt") as algoarttab:
        algoart_row = gr.Interface(
            fn=create_template,
            inputs=[gr.Slider(0, 10, step=0.1, value=5),
                    gr.Slider(0, 1.0, step=0.1, value=0.8),
                    gr.Slider(0, 0.99, step=0.05, value=0.9),
                    gr.Slider(10, 100, step=5, value=60),
                    gr.Slider(500, 10000, step=500, value=5000),
                    gr.Dropdown([0.01, 0.1], value=0.01),
                    ],
            outputs="image")
    with gr.Tab("Stable Diffusion"):
        gr.Interface(
            fn=run_stable_diff,
            inputs=['text', algoart_row.output_components[0]],
            outputs="image")


if __name__ == "__main__":
    demo.launch(show_api=False, server_name="0.0.0.0", server_port=8000)
