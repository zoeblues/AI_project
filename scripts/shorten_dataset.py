import os
import pandas as pd
import shutil
from os.path import join as pjoin


def shorten_dataset(input_dir, output_dir, base_dir="data", downsample_weight=0.2, out_file='training-mini.csv'):
	# Dataset
	dataset_csv = []
	
	input_dir = os.path.join(base_dir, input_dir)
	output_dir = os.path.join(base_dir, output_dir)
	for class_dir in os.listdir(input_dir):
		if not os.path.isdir(pjoin(input_dir, class_dir)):
			continue
		# Recreate in output
		os.makedirs(pjoin(output_dir, class_dir), exist_ok=True)
		input_class_dir = os.path.join(input_dir, class_dir)
		output_class_dir = os.path.join(output_dir, class_dir)
		
		file_count = sum(1 for file in os.listdir(input_class_dir) if any([file.endswith(extension) for extension in [".jpg", ".png", ".jpeg"]]) and not file.startswith("."))
		smaller_count = int(file_count * downsample_weight)
		
		i = 0
		for filename in os.listdir(input_class_dir):
			if not any([filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg"]]) or filename.startswith("."):
				continue
			
			input_file_path = os.path.join(input_class_dir, filename)
			output_file_path = os.path.join(output_class_dir, filename)
			shutil.copy(input_file_path, output_file_path)
			dataset_csv.append({"label": class_dir, "path": output_file_path})
			
			i += 1
			if i >= smaller_count:
				break
	
	pd.DataFrame(dataset_csv).to_csv(os.path.join(base_dir, out_file), index=False)


if __name__ == '__main__':
	shorten_dataset('resized_images', 'less_images')
