import kagglehub
import shutil
import os


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


if __name__ == '__main__':
	# WARNING!!! THE DATASET IS 20GB
	data_path = download()
	remove_original(data_path)
