import os
import pandas as pd


def create_dataset_csv(data_folder, data_path='data', file_name='training.csv'):
	# Dataset
	dataset_csv = []
	
	dataset_dir = os.path.join(data_path, data_folder)
	for class_dir in os.listdir(dataset_dir):
		class_dir_path = os.path.join(dataset_dir, class_dir)
		# Make sure this is a directory
		if not os.path.isdir(class_dir_path):
			continue
		for filename in os.listdir(class_dir_path):
			# Make sure file is an image
			if not any([filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg"]]):
				continue
			# Create path and add to dataset
			class_file_path = os.path.join(class_dir_path, filename)
			dataset_csv.append({"label": class_dir, "path": class_file_path})
	# With pandas save the dataset information
	pd.DataFrame(dataset_csv).to_csv(os.path.join(data_path, file_name), index=False)


if __name__ == '__main__':
	create_dataset_csv('resized_images', data_path='data', file_name='diffusion_training.csv')
