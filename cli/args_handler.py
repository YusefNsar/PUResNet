import argparse
import os
from openbabel import pybel


class Args:
  file_format: str
  multi: int
  model_path: str
  input_path: str
  output_path: str
  output_format: str
  gpu: str


class ArgsHandler:
  @staticmethod
  def parse() -> Args:
    parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--file_format', '-ftype', required=True, type=str,
                        help='File Format of Protein Structure like: mol2,pdb..etc. All file format supported by Open Babel is supported')
    parser.add_argument('--multi', '-m', required=True, type=int,
                        help='Multi 0 is for single protein structure. Multi 1 is for multiple protein structure')
    parser.add_argument('--model_path', '-mpath', required=True, type=str,
                        help='Provide models absolute or relative path of model')
    parser.add_argument('--input_path', '-i', required=True, type=str,
                        help='For multi 0 provide absolute or relative path for protein structure. For multi 1 provide absolute or relative path for folder containing protein structure')
    parser.add_argument('--output_format', '-otype', required=False, type=str, default='mol2',
                        help='Provide the output format for predicted binding side. All formats supported by Open Babel')
    parser.add_argument('--output_path', '-o', required=False,
                        type=str, default='output', help='path to model output')
    parser.add_argument('--gpu', '-gpu', required=False, type=str,
                        help='Provide GPU device if you want to use GPU like: 0 or 1 or 2 etc.')

    return parser.parse_args()

  @staticmethod
  def validate(args: Args) -> bool:
    if args.multi not in [0, 1]:
      raise ValueError('Please Enter Valid value for multi: 0 or 1')
    elif args.multi == 0:
      if not os.path.isfile(args.input_path):
        raise FileNotFoundError('Input File Not Found')
    elif args.multi == 1:
      if not os.path.isdir(args.input_path):
        raise FileNotFoundError('Input Folder Not Found')

    if not os.path.exists(args.output_path):
      os.mkdir(args.output_path)

    if args.file_format not in pybel.informats.keys():
      raise ValueError('Enter Valid File Format {}'.format(pybel.informats))

    if args.output_format not in pybel.outformats.keys():
      raise ValueError(
        'Enter Valid Output Format {}'.format(pybel.outformats))

    if args.gpu:
      os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
      os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
