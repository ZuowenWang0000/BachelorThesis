import argparse

import os
import utilities


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Resource request')
  parser.add_argument(
    '--repo_dir',
    type=str,
    default='tmp/',
    help='repo directory')
  parser.add_argument(
    '--local_json_dir_name',
    type=str,
    default='local_json_files',
    help='directory with local json files')
  parser.add_argument(
    '--output_dir',
    type=str,
    default='tmp/',
    help='directory where to save output file')
  parser.add_argument(
    '--output_filename',
    type=str,
    default='all_experiments.json',
    help='name for output json file containing all experiments')
  args = parser.parse_args()

  local_json_dir = os.path.join(args.repo_dir, args.local_json_dir_name)
  file_list = [os.path.join(local_json_dir, f)
               for f in os.listdir(local_json_dir)
               if os.path.isfile(os.path.join(local_json_dir, f))]
  output_path = os.path.join(args.output_dir, args.output_filename)
  utilities.concatenate_json_files(file_list, output_path)
