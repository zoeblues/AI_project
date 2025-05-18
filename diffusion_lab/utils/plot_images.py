import os
from time import sleep

import numpy as np
from PIL import Image
from os.path import join as pjoin
from matplotlib import pyplot as plt
from moviepy import VideoFileClip, ImageSequenceClip


def show_save_images(image_array, save_path="results", title="Image.jpg", show=True):
	os.makedirs(save_path, exist_ok=True)
	
	n_rows = len(image_array[0])
	n_columns = len(image_array)
	
	bgc = Image.new("RGB", (64 * n_columns, 64 * n_rows), color=(255, 255, 255)).convert("RGB")
	
	for c in range(n_columns):
		for r in range(n_rows):
			bgc.paste(image_array[c][r], (64 * c, 64 * r))
	
	if show:
		bgc.show(title=title)
	bgc.save(pjoin(save_path, title))


def save_gif(images, save_path="results", title="GIF.gif"):
	os.makedirs(save_path, exist_ok=True)
	
	# Save images as GIF
	images[0].save(
		pjoin(save_path, title),
		save_all=True,
		append_images=images[1:],
		duration=0,  # milliseconds
		loop=1
	)
	
	# Save as MP4 for easier sharing
	img_np_array = [np.array(img) for img in images]
	clip = ImageSequenceClip(img_np_array, fps=24)
	clip.write_videofile(pjoin(save_path, f"{title}.mp4"), codec="libx264", fps=24)


def plot_lines(x_values, y_values, line_labels, x_label="X-Label", y_label="Y-Label", title="Title", save_path="results", show=True):
	# Add line values to the plot
	for i in range(len(x_values)):
		plt.plot(x_values[i], y_values[i], label=line_labels[i])
	
	# Adding labels and legend
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.legend()
	
	plt.savefig(pjoin(save_path, f"{title}.png"), dpi=300)
	
	if show:
		plt.show()
	
	plt.close()
