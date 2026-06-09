#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a ConvRot-trained LoRA checkpoint back to regular activation coordinates.'
    )
    parser.add_argument('input', type=Path, help='Input LoRA .safetensors file.')
    parser.add_argument('output', type=Path, help='Output LoRA .safetensors file.')
    parser.add_argument(
        '--group-size',
        type=int,
        default=256,
        help='ConvRot group size used during training. Default: 256.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from utils.convrot import build_hadamard, rotate_weight

    if args.input == args.output:
        raise ValueError('Input and output paths must be different.')
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    metadata = None
    with safe_open(args.input, framework='pt', device='cpu') as f:
        metadata = f.metadata()

    state_dict = load_file(args.input, device='cpu')
    hadamards = {}
    converted = 0

    for key, tensor in list(state_dict.items()):
        if not _is_lora_a_weight(key, tensor):
            continue
        if tensor.shape[1] % args.group_size != 0:
            raise ValueError(
                f'{key} has in_features={tensor.shape[1]}, which is not divisible by group size {args.group_size}'
            )
        h = hadamards.get(args.group_size)
        if h is None:
            h = build_hadamard(args.group_size, device='cpu', dtype=torch.float32)
            hadamards[args.group_size] = h
        state_dict[key] = rotate_weight(tensor.to(torch.float32), h, args.group_size).to(tensor.dtype)
        converted += 1

    if converted == 0:
        raise RuntimeError('No lora_A weight tensors were found to unrotate.')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, args.output, metadata=metadata)
    print(f'Unrotated {converted} LoRA A tensors')
    print(f'Wrote {args.output}')


def _is_lora_a_weight(key, tensor):
    return tensor.ndim == 2 and 'lora_A' in key and key.endswith('.weight')


if __name__ == '__main__':
    main()
