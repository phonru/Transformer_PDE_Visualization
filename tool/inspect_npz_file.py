import numpy as np
import argparse

def main():
	parser = argparse.ArgumentParser(description="Inspect the keys and shapes in a .npz file.")
	parser.add_argument('--file', type=str,
					 	default='data/hidden_states/dataset_3/sample_0.npz',
						help='Path to the .npz file to inspect.')
	args = parser.parse_args()

	file_path = args.file
	try:
		data = np.load(file_path)
	except Exception as e:
		print(f"Error loading file: {e}")
		return

	print("Keys and shapes in npz file:")
	for key in data.files:
		arr = data[key]
		print(f"{key}: {arr.shape}")

if __name__ == "__main__":
	main()