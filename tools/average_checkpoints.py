"""将两个同结构的MMSegmentation checkpoint平均为一个checkpoint。

示例：
    python tools/average_checkpoints.py model_a.pth model_b.pth merged.pth

生成的merged.pth只包含一套模型参数，推理时只加载一个模型。
"""

import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='平均两个同结构checkpoint，生成一个新checkpoint')
    parser.add_argument('checkpoint_a', help='第一个checkpoint')
    parser.add_argument('checkpoint_b', help='第二个checkpoint')
    parser.add_argument('output', help='合并后的checkpoint保存路径')
    parser.add_argument(
        '--weight-a',
        type=float,
        default=0.5,
        help='第一个checkpoint的权重，默认0.5；第二个权重为1-weight-a')
    return parser.parse_args()


def get_state_dict(checkpoint, path):
    if not isinstance(checkpoint, dict):
        raise TypeError(f'{path}不是有效的checkpoint字典')

    state_dict = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f'{path}中没有有效的state_dict')
    return state_dict


def load_checkpoint(path):
    """兼容新旧PyTorch的checkpoint加载方式。"""
    try:
        return torch.load(
            str(path), map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location='cpu')


def main():
    args = parse_args()
    if not 0.0 <= args.weight_a <= 1.0:
        raise ValueError('--weight-a必须在[0, 1]范围内')

    path_a = Path(args.checkpoint_a)
    path_b = Path(args.checkpoint_b)
    output_path = Path(args.output)

    if not path_a.is_file():
        raise FileNotFoundError(f'找不到第一个checkpoint: {path_a}')
    if not path_b.is_file():
        raise FileNotFoundError(f'找不到第二个checkpoint: {path_b}')

    print(f'加载模型A: {path_a}')
    checkpoint_a = load_checkpoint(path_a)
    print(f'加载模型B: {path_b}')
    checkpoint_b = load_checkpoint(path_b)

    state_a = get_state_dict(checkpoint_a, path_a)
    state_b = get_state_dict(checkpoint_b, path_b)

    keys_a = set(state_a.keys())
    keys_b = set(state_b.keys())
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        raise KeyError(
            '两个checkpoint的参数名称不一致，不能直接平均。\n'
            f'仅模型A存在: {only_a[:10]}\n'
            f'仅模型B存在: {only_b[:10]}')

    weight_a = args.weight_a
    weight_b = 1.0 - weight_a
    merged_state = {}
    copied_non_float = 0

    for key in state_a:
        tensor_a = state_a[key]
        tensor_b = state_b[key]

        if not isinstance(tensor_a, torch.Tensor) or not isinstance(
                tensor_b, torch.Tensor):
            raise TypeError(f'参数{key}不是Tensor，不能平均')
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(
                f'参数{key}形状不一致: {tensor_a.shape} 与 {tensor_b.shape}')

        if tensor_a.is_floating_point():
            # 使用float32计算，防止半精度参数平均时损失精度。
            merged = (
                tensor_a.float().mul(weight_a) +
                tensor_b.float().mul(weight_b))
            merged_state[key] = merged.to(dtype=tensor_a.dtype)
        else:
            # num_batches_tracked等整数缓存不能做浮点平均。
            if not torch.equal(tensor_a, tensor_b):
                print(f'[WARN] 非浮点参数不同，保留模型A: {key}')
            merged_state[key] = tensor_a.clone()
            copied_non_float += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_checkpoint = {
        'meta': checkpoint_a.get('meta', {}),
        'state_dict': merged_state,
    }
    torch.save(merged_checkpoint, str(output_path))

    print(f'参数数量: {len(merged_state)}')
    print(f'直接复制的非浮点参数: {copied_non_float}')
    print(f'模型A权重: {weight_a:.3f}')
    print(f'模型B权重: {weight_b:.3f}')
    print(f'已生成单模型checkpoint: {output_path}')


if __name__ == '__main__':
    main()
