import kagglehub
import shutil
import os
from img_data_to_csv import create_dataset_csv


def download():
	# Make sure directory exists
	os.makedirs("./data", exist_ok=True)
	
	# Download latest version
	path = kagglehub.dataset_download("utkarshsaxenadn/animal-image-classification-dataset")
	shutil.copytree(path, "./data", dirs_exist_ok=True)
	
	print("Path to dataset files:", path)
	return path


def remove_original(path):
	if os.path.exists(path):
		shutil.rmtree(path)
		print(f"Deleted: {path}")
	else:
		print("Folder does not exist.")


def remove_not_needed_data():
	# Removing useless data, original training steps
	if os.path.exists("./data/TFRecords Data"):
		shutil.rmtree("./data/TFRecords Data")
		print("Deleted: ./data/TFRecords Data")
	# Removing data that were designed for classification generalization testing,
	# not needed for image generation
	if os.path.exists("./data/Interesting Data"):
		shutil.rmtree("./data/Interesting Data")
		print("Deleted: ./data/Interesting Data")
	# Remove that was forcibly rotated for classification generalization,
	# Can be done by oneself if necessary
	if os.path.exists("./data/Train Augmented"):
		shutil.rmtree("./data/Train Augmented")
		print("Deleted: ./data/Train Augmented")


def reformat_simplify_structure():
	# Target dir, set and make sure it exists
	target_dir = "./data/diffusion_training/"
	os.makedirs(target_dir, exist_ok=True)
	# Set source dirs
	source_dirs = [
		'./data/Testing Data/Testing Data/',
		'./data/Training Data/Training Data/',
		'./data/Validation Data/Validation Data/'
	]
	
	for source_dir in source_dirs:
		if not os.path.exists(source_dir):
			print(f"Source directory does not exist: {source_dir}")
			continue
		# Move files into target dirs
		for sub_dir in os.listdir(source_dir):
			sub_dir_path = os.path.join(source_dir, sub_dir)
			# Only consider directories in source directories
			if not os.path.isdir(sub_dir_path):
				continue
			# Make sure class dictionary exits in target
			os.makedirs(os.path.join(target_dir, sub_dir), exist_ok=True)
			# Iterate over files and copy
			for filename in os.listdir(sub_dir_path):
				# Make sure file is an image
				if not any([filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg"]]):
					continue
				source_file_path = os.path.join(sub_dir_path, filename)
				# Format name to not include SPACE chara
				formatted_filename = filename.replace(" ", "_")
				target_file_path = os.path.join(target_dir, sub_dir, formatted_filename)
				shutil.copy2(source_file_path, target_file_path)
		if os.path.exists(source_dir):
			shutil.rmtree(source_dir)
	
	# Final Removal
	if os.path.exists("./data/Testing Data/"):
		shutil.rmtree("./data/Testing Data/")
	if os.path.exists("./data/Training Data/"):
		shutil.rmtree("./data/Training Data/")
	if os.path.exists("./data/Validation Data/"):
		shutil.rmtree("./data/Validation Data/")


if __name__ == '__main__':
	# WARNING!!! THE DATASET IS 20GB
	data_path = download()
	remove_original(data_path)
	remove_not_needed_data()
	reformat_simplify_structure()
	create_dataset_csv('diffusion_training', 'data', file_name='diffusion_training.csv')
