import h5py
import os
import argparse

def find_h5_files(root_dir):

    h5_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.h5'):
                h5_files.append(os.path.join(dirpath, filename))
    return h5_files

def print_h5_structure(g, prefix=""):

    for key in g.keys():
        item = g[key]
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            print(f"    {path}: shape={item.shape}, dtype={item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"    {path}: Group")
            print_h5_structure(item, path)

def main():
    parser = argparse.ArgumentParser(description="Recursively inspect the structure of all .h5 files in specified directories or files.")
    parser.add_argument('--paths', nargs='+',
                        default=[''],
                        help='One or more directories or .h5 file paths to inspect.')
    args = parser.parse_args()

    file_paths = []
    for d in args.paths:
        if os.path.isdir(d):
            file_paths.extend(find_h5_files(d))
        elif os.path.isfile(d) and d.endswith('.h5'):
            file_paths.append(d)
        else:
            print(f'Warning: {d} is not a valid directory or .h5 file')

    if not file_paths:
        print('No .h5 files found')
        return

    for file_path in file_paths:
        print(f'\nFile: {file_path}')
        try:
            with h5py.File(file_path, 'r') as f:
                print('  Keys:')
                print_h5_structure(f)
        except Exception as e:
            print(f'  [Error] {e}')

if __name__ == "__main__":
    main()