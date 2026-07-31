"""Convert a PyTorch checkpoint to a pure-numpy pickle loadable from Jittor.

Run inside the p2r-torch env:
    python tools/convert_torch_weights.py <input.pth> <output.pkl>

On the Jittor side:
    import pickle, jittor as jt
    params = pickle.load(open('weights/ted.pkl', 'rb'))
    model.load_parameters({k: jt.array(v) for k, v in params.items()})

Notes:
- Handles checkpoints wrapped as {'state_dict': ...} or {'model': ...}.
- Conv2d / Linear weight layouts match between torch and Jittor; no
  transposition is done here. Modules that need repacking (e.g. torch
  nn.MultiheadAttention in_proj_weight -> separate q/k/v Linears) must be
  handled by a model-specific converter on top of this dump.
- Non-tensor entries (ints, version tags) are dropped with a warning.
"""
import pickle
import sys

import torch


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    sd = torch.load(src, map_location='cpu')
    if isinstance(sd, dict):
        sd = sd.get('state_dict', sd.get('model', sd))
    out = {}
    dropped = []
    for k, v in sd.items():
        if hasattr(v, 'detach'):
            out[k] = v.detach().cpu().numpy()
        else:
            dropped.append(k)
    with open(dst, 'wb') as f:
        pickle.dump(out, f)
    print(f'{len(out)} tensors -> {dst}')
    if dropped:
        print(f'dropped non-tensor keys: {dropped}')


if __name__ == '__main__':
    main()
