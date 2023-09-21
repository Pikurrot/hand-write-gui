import os, re

def ensure_dir(dir_name: str) -> None:
	'''Create the directory if it doesn't exist'''
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)

def get_save_file(save_dir: str, file_format: str = '.safetensors') -> str:
	'''Returns the next available file name in the root_dir with the specified file_format'''
	# Determine the highest counter already used in file names
	highest_counter = -1
	for existing_file in os.listdir(save_dir):
		if existing_file.endswith(file_format):
			match = re.match(r'(\d+)', existing_file)
			if match:
				highest_counter = max(highest_counter, int(match.group(1)))

	# Determine the next available file name with counter
	next_counter = highest_counter + 1
	return os.path.join(save_dir, f'model_{next_counter}{file_format}')