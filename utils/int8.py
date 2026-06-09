import torch
from torch import nn
import torch.nn.functional as F

from utils.convrot import build_hadamard, rotate_activation, rotate_weight
from utils.int8_kernel_triton import triton_int8_linear_per_row


def is_int8_dtype(dtype):
    return dtype is torch.int8 or dtype == torch.int8


class Int8Linear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        compute_dtype=torch.bfloat16,
        convrot_group_size=None,
    ):
        super().__init__(in_features, out_features, bias=bias, dtype=compute_dtype)
        del self._parameters['weight']
        self.register_buffer('weight', torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer('weight_scale', torch.empty((out_features,), dtype=torch.float32))
        self.register_buffer('convrot_h', None)
        self.compute_dtype = compute_dtype
        self.convrot_group_size = convrot_group_size
        self.rotate_input_in_forward = True

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear, compute_dtype=None, use_convrot=False, convrot_group_size=256):
        compute_dtype = compute_dtype or _infer_compute_dtype(linear)
        convrot_group_size = _get_convrot_group_size(linear.in_features, use_convrot, convrot_group_size)
        new_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=compute_dtype,
            convrot_group_size=convrot_group_size,
        )
        weight = linear.weight
        if weight.__class__.__name__ == 'QuantizedTensor':
            weight = weight.dequantize()
        weight = weight.detach().to(torch.float32)
        if convrot_group_size is not None:
            h = build_hadamard(convrot_group_size, device=weight.device, dtype=torch.float32)
            weight = rotate_weight(weight, h, convrot_group_size)
            new_linear.convrot_h = h.to(device=linear.weight.device)
        scale = weight.abs().amax(dim=1).clamp(min=1e-30) / 127.0
        quantized_weight = torch.round(weight / scale[:, None]).clamp(-128, 127).to(torch.int8)

        new_linear.weight = quantized_weight.to(device=linear.weight.device)
        new_linear.weight_scale = scale.to(device=linear.weight.device)
        if linear.bias is not None:
            new_linear.bias.data.copy_(linear.bias.detach().to(device=linear.bias.device, dtype=compute_dtype))
            new_linear.bias.requires_grad_(False)
        return new_linear

    def forward(self, input):
        bias = self.bias
        if self.rotate_input_in_forward and self.convrot_group_size is not None:
            input = rotate_activation(input, self.convrot_h, self.convrot_group_size)
        if input.is_cuda:
            compute_dtype = input.dtype if input.dtype.is_floating_point else self.compute_dtype
            return triton_int8_linear_per_row(input, self.weight, self.weight_scale, bias, compute_dtype)

        weight = self.weight.to(dtype=input.dtype) * self.weight_scale.to(dtype=input.dtype)[:, None]
        return F.linear(input, weight, bias)

    def extra_repr(self):
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, weight_dtype=int8, weight_scale=per_row, '
            f'convrot_group_size={self.convrot_group_size}'
        )


def quantize_linear_modules(
    module,
    compute_dtype=torch.bfloat16,
    keep_in_high_precision=(),
    prefix='',
    use_convrot=False,
    convrot_group_size=256,
):
    for name, child in list(module.named_children()):
        child_prefix = f'{prefix}.{name}' if prefix else name
        if isinstance(child, nn.Linear) and not isinstance(child, Int8Linear):
            if _matches_keep_in_high_precision(child, child_prefix, keep_in_high_precision):
                continue
            module._modules[name] = Int8Linear.from_linear(
                child,
                compute_dtype=compute_dtype,
                use_convrot=use_convrot,
                convrot_group_size=convrot_group_size,
            )
        else:
            quantize_linear_modules(
                child,
                compute_dtype=compute_dtype,
                keep_in_high_precision=keep_in_high_precision,
                prefix=child_prefix,
                use_convrot=use_convrot,
                convrot_group_size=convrot_group_size,
            )


def enable_convrot_lora_activation_rotation(module):
    for child in module.modules():
        base_layer = _get_base_layer(child)
        if not isinstance(base_layer, Int8Linear) or base_layer.convrot_group_size is None:
            continue
        if getattr(child, '_convrot_lora_activation_hooked', False):
            continue
        base_layer.rotate_input_in_forward = False
        child.register_forward_pre_hook(_convrot_lora_activation_pre_hook)
        child._convrot_lora_activation_hooked = True


def _infer_compute_dtype(linear):
    if linear.bias is not None and linear.bias.dtype.is_floating_point:
        return linear.bias.dtype
    if linear.weight.dtype.is_floating_point:
        return linear.weight.dtype
    return torch.bfloat16


def _matches_keep_in_high_precision(linear, module_name, keep_in_high_precision):
    names = [module_name, f'{module_name}.weight']
    if linear.bias is not None:
        names.append(f'{module_name}.bias')
    return any(keyword in name for keyword in keep_in_high_precision for name in names)


def _get_convrot_group_size(in_features, use_convrot, group_size):
    if not use_convrot:
        return None
    if in_features % group_size != 0:
        return None
    return group_size


def _get_base_layer(module):
    if isinstance(module, Int8Linear):
        return None
    if hasattr(module, 'get_base_layer'):
        return module.get_base_layer()
    return getattr(module, 'base_layer', None)


def _convrot_lora_activation_pre_hook(module, args):
    if len(args) == 0:
        return args
    base_layer = _get_base_layer(module)
    if not isinstance(base_layer, Int8Linear) or base_layer.convrot_group_size is None:
        return args
    x = rotate_activation(args[0], base_layer.convrot_h, base_layer.convrot_group_size)
    return (x, *args[1:])
