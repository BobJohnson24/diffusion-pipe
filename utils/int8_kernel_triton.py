import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import logging

# =============================================================================
# Shape Logger  (prints each unique call signature exactly once)
# =============================================================================

_shape_log = logging.getLogger("int8_kernel")
if not _shape_log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[INT8-KERNEL] %(message)s"))
    _shape_log.addHandler(_h)
    _shape_log.setLevel(logging.DEBUG)
    _shape_log.propagate = False

_logged_shapes: set = set()   # Dedup: only log new shapes

def _log_call(tag: str, x_shape_orig, M: int, N: int, K: int,
              weight_dtype, scale_type: str, has_bias: bool,
              compute_dtype, x_dtype):
    key = (tag, x_shape_orig, M, N, K, scale_type, has_bias, str(compute_dtype), str(x_dtype))
    if key in _logged_shapes:
        return
    _logged_shapes.add(key)
    _shape_log.debug(
        f"{tag} | x_orig={x_shape_orig} x_dtype={x_dtype} "
        f"| M={M} N={N} K={K} "
        f"| weight_dtype={weight_dtype} scale={scale_type} "
        f"| bias={'yes' if has_bias else 'no'} "
        f"| out_dtype={compute_dtype}"
    )

# =============================================================================
# Kernel 1: Fused Row-wise Quantization (FP16/BF16 -> INT8 + Scale)
# =============================================================================

@triton.jit
def _quantize_rowwise_kernel(
    x_ptr,      # Input pointer (FP16/BF16)
    y_ptr,      # Output pointer (INT8)
    s_ptr,      # Scale pointer (FP32)
    n_elements, # Number of columns
    BLOCK_SIZE: tl.constexpr,
):
    # Row index we are processing
    row_idx = tl.program_id(0)
    
    # Pointers to the start of the row
    x_row_ptr = x_ptr + row_idx * n_elements
    y_row_ptr = y_ptr + row_idx * n_elements
    
    # 1. Compute Max Abs Value for the row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
    
    # Absolute value
    abs_x = tl.abs(x)
    
    # Reduction to find max
    max_val = tl.max(abs_x, axis=0)
    
    # 2. Compute Scale
    scale = tl.maximum(max_val / 127.0, 1e-30)
    
    # 3. Quantize
    q_f = x / scale
    
    # Round and Clamp
    q_i = libdevice.rint(q_f).to(tl.int32)
    q_i = tl.clamp(q_i, -128.0, 127.0)
    
    # 4. Store
    tl.store(y_row_ptr + offsets, q_i.to(tl.int8), mask=mask)
    tl.store(s_ptr + row_idx, scale.to(tl.float32))

def triton_quantize_rowwise(x: torch.Tensor):
    """
    Input: [Batch, Dim] (float16/bfloat16/float32)
    Output: [Batch, Dim] (int8), [Batch, 1] (float32)
    """
    rows, cols = x.shape
    y = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    
    BLOCK_SIZE = triton.next_power_of_2(cols)
    if BLOCK_SIZE < 128: BLOCK_SIZE = 128
    
    grid = (rows,)
    _quantize_rowwise_kernel[grid](x, y, s, cols, BLOCK_SIZE=BLOCK_SIZE)
    return y, s


# =============================================================================
# Kernel 2: Fused Row-wise Quantization for Backward Pass (absorbs weight scale)
# =============================================================================

@triton.jit
def _quantize_bwd_rowwise_kernel(
    grad_y_ptr,
    w_scale_ptr,
    out_int8_ptr,
    out_scale_ptr,
    n_elements,
    stride_gym, stride_gyn,
    PER_ROW_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    grad_y_row_ptr = grad_y_ptr + row_idx * stride_gym
    out_int8_row_ptr = out_int8_ptr + row_idx * n_elements
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load grad_output
    gy = tl.load(grad_y_row_ptr + offsets * stride_gyn, mask=mask, other=0.0)
    
    # Fold weight scale in dynamically before quantizing
    if PER_ROW_SCALE:
        ws = tl.load(w_scale_ptr + offsets, mask=mask, other=1.0)
        gy = gy * ws
    else:
        ws = tl.load(w_scale_ptr)
        gy = gy * ws
        
    abs_gy = tl.abs(gy)
    max_val = tl.max(abs_gy, axis=0)
    scale = tl.maximum(max_val / 127.0, 1e-30)
    
    q_f = gy / scale
    q_i = libdevice.rint(q_f).to(tl.int32)
    q_i = tl.clamp(q_i, -128.0, 127.0)
    
    tl.store(out_int8_row_ptr + offsets, q_i.to(tl.int8), mask=mask)
    tl.store(out_scale_ptr + row_idx, scale.to(tl.float32))

def triton_quantize_bwd_rowwise(grad_y: torch.Tensor, weight_scale: torch.Tensor, per_row_scale: bool):
    rows, cols = grad_y.shape
    out_int8 = torch.empty_like(grad_y, dtype=torch.int8)
    out_scale = torch.empty((rows, 1), device=grad_y.device, dtype=torch.float32)
    
    BLOCK_SIZE = triton.next_power_of_2(cols)
    if BLOCK_SIZE < 128: BLOCK_SIZE = 128
    
    grid = (rows,)
    w_scale = weight_scale.contiguous() if per_row_scale else weight_scale
    
    _quantize_bwd_rowwise_kernel[grid](
        grad_y, w_scale, out_int8, out_scale, cols,
        grad_y.stride(0), grad_y.stride(1),
        PER_ROW_SCALE=per_row_scale, BLOCK_SIZE=BLOCK_SIZE
    )
    return out_int8, out_scale

@triton.jit
def _quantize_bwd_weight_rowwise_kernel(
    grad_y_ptr,
    x_scale_ptr,
    out_int8_ptr,
    out_scale_ptr,
    M: tl.constexpr,
    stride_gym, stride_gyn,
    stride_outm, stride_outn,
    BLOCK_M: tl.constexpr,
):
    """ Quantizes transpose of grad_y while absorbing x_scale for computing grad_weight """
    n_idx = tl.program_id(0)
    
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    
    gy_ptrs = grad_y_ptr + offs_m * stride_gym + n_idx * stride_gyn
    gy = tl.load(gy_ptrs, mask=mask_m, other=0.0)
    
    xs_ptrs = x_scale_ptr + offs_m
    xs = tl.load(xs_ptrs, mask=mask_m, other=1.0)
    
    gy = gy * xs
    
    abs_gy = tl.abs(gy)
    max_val = tl.max(abs_gy, axis=0)
    scale = tl.maximum(max_val / 127.0, 1e-30)
    
    q_f = gy / scale
    q_i = libdevice.rint(q_f).to(tl.int32)
    q_i = tl.clamp(q_i, -128.0, 127.0)
    
    out_ptrs = out_int8_ptr + n_idx * stride_outm + offs_m * stride_outn
    tl.store(out_ptrs, q_i.to(tl.int8), mask=mask_m)
    tl.store(out_scale_ptr + n_idx, scale.to(tl.float32))

def triton_quantize_bwd_weight_rowwise(grad_y: torch.Tensor, x_scale: torch.Tensor):
    M, N = grad_y.shape
    out_int8 = torch.empty((N, M), device=grad_y.device, dtype=torch.int8)
    out_scale = torch.empty((N, 1), device=grad_y.device, dtype=torch.float32)
    
    BLOCK_M = triton.next_power_of_2(M)
    if BLOCK_M < 128: BLOCK_M = 128
    
    grid = (N,)
    _quantize_bwd_weight_rowwise_kernel[grid](
        grad_y, x_scale, out_int8, out_scale,
        M, grad_y.stride(0), grad_y.stride(1),
        out_int8.stride(0), out_int8.stride(1),
        BLOCK_M=BLOCK_M
    )
    return out_int8, out_scale

# =============================================================================
# Kernel 3 & 4: INT8 GEMM + Fused Dequantization Epilogue (Forward)
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _int8_matmul_dequant_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    scale_a = tl.load(a_scale_ptr + offs_am)
    scale_b = tl.load(b_scale_ptr) 

    c = accumulator.to(tl.float32)
    total_scale = scale_a[:, None] * scale_b
    c = c * total_scale

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn)
        c = c + bias[None, :]

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _int8_matmul_dequant_per_row_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    scale_a = tl.load(a_scale_ptr + offs_am)
    scale_b = tl.load(b_scale_ptr + offs_bn)

    c = accumulator.to(tl.float32)
    total_scale = scale_a[:, None] * scale_b[None, :]
    c = c * total_scale

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn)
        c = c + bias[None, :]

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# =============================================================================
# Autograd Function Wrapper
# =============================================================================

class TritonInt8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, weight_scale, bias=None, compute_dtype=torch.float16, per_row_scale=False):
        x_shape_orig = x.shape
        x_2d = x.reshape(-1, x_shape_orig[-1])
        M, K = x_2d.shape
        N = weight.shape[0]

        # _log_call("per_row" if per_row_scale else "tensorwise", x_shape_orig, M, N, K,
        #           weight.dtype, f"per_row[{weight_scale.shape}]" if per_row_scale else "scalar", 
        #           bias is not None, compute_dtype, x.dtype)

        # Forward dynamic quant
        x_int8, x_scale = triton_quantize_rowwise(x_2d)

        output = torch.empty((M, N), device=x.device, dtype=compute_dtype)

        if not isinstance(weight_scale, torch.Tensor):
            weight_scale = torch.tensor([weight_scale], device=x.device, dtype=torch.float32)
        elif weight_scale.numel() == 1 and not per_row_scale:
            weight_scale = weight_scale.reshape(1)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
        has_bias = bias is not None
        bias_ptr = bias if has_bias else x

        if per_row_scale:
            ws = weight_scale.reshape(N).contiguous()
            _int8_matmul_dequant_per_row_kernel[grid](
                a_ptr=x_int8, b_ptr=weight, c_ptr=output,
                a_scale_ptr=x_scale, b_scale_ptr=ws, bias_ptr=bias_ptr,
                M=M, N=N, K=K,
                stride_am=x_int8.stride(0), stride_ak=x_int8.stride(1),
                stride_bk=weight.stride(1), stride_bn=weight.stride(0),
                stride_cm=output.stride(0), stride_cn=output.stride(1),
                HAS_BIAS=has_bias
            )
        else:
            _int8_matmul_dequant_kernel[grid](
                a_ptr=x_int8, b_ptr=weight, c_ptr=output,
                a_scale_ptr=x_scale, b_scale_ptr=weight_scale, bias_ptr=bias_ptr,
                M=M, N=N, K=K,
                stride_am=x_int8.stride(0), stride_ak=x_int8.stride(1),
                stride_bk=weight.stride(1), stride_bn=weight.stride(0),
                stride_cm=output.stride(0), stride_cn=output.stride(1),
                HAS_BIAS=has_bias
            )

        # Stash for fused backward
        ctx.save_for_backward(x_int8, x_scale, weight, weight_scale)
        ctx.per_row_scale = per_row_scale
        ctx.x_shape_orig = x_shape_orig
        ctx.has_bias = has_bias
        ctx.compute_dtype = compute_dtype

        return output.reshape(x_shape_orig[:-1] + (N,))

    @staticmethod
    def backward(ctx, grad_output):
        x_int8, x_scale, weight, weight_scale = ctx.saved_tensors
        per_row_scale = ctx.per_row_scale
        x_shape_orig = ctx.x_shape_orig
        
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        M, N = grad_output_2d.shape
        K = weight.shape[1]

        grad_x = None
        grad_weight = None
        grad_weight_scale = None
        grad_bias = None

        # 1. Backward pass for grad_x (requires quantizing grad_y)
        if ctx.needs_input_grad[0]:
            # Scale absorbing quantize step
            grad_y_int8, grad_y_scale = triton_quantize_bwd_rowwise(grad_output_2d, weight_scale, per_row_scale)
            
            grad_x_2d = torch.empty((M, K), device=grad_output.device, dtype=ctx.compute_dtype)
            dummy_b_scale = torch.tensor([1.0], device=grad_output.device, dtype=torch.float32)
            
            # Repurpose the GEMM kernel: (M, N) * (N, K) -> (M, K)
            grid_x = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(K, META['BLOCK_N']), )
            _int8_matmul_dequant_kernel[grid_x](
                a_ptr=grad_y_int8, b_ptr=weight, c_ptr=grad_x_2d,
                a_scale_ptr=grad_y_scale, b_scale_ptr=dummy_b_scale, bias_ptr=grad_y_int8,
                M=M, N=K, K=N,
                stride_am=grad_y_int8.stride(0), stride_ak=grad_y_int8.stride(1),
                stride_bk=weight.stride(0), stride_bn=weight.stride(1),
                stride_cm=grad_x_2d.stride(0), stride_cn=grad_x_2d.stride(1),
                HAS_BIAS=False
            )
            grad_x = grad_x_2d.reshape(x_shape_orig)

        # 2. Backward pass for grad_weight (typically useful for QAT or float master weights updates)
        if ctx.needs_input_grad[1]:
            grad_y_t_int8, grad_y_t_scale = triton_quantize_bwd_weight_rowwise(grad_output_2d, x_scale)
            
            grad_weight = torch.empty((N, K), device=grad_output.device, dtype=ctx.compute_dtype)
            dummy_b_scale = torch.tensor([1.0], device=grad_output.device, dtype=torch.float32)
            
            # Repurpose the GEMM kernel: (N, M) * (M, K) -> (N, K)
            grid_w = lambda META: (triton.cdiv(N, META['BLOCK_M']) * triton.cdiv(K, META['BLOCK_N']), )
            _int8_matmul_dequant_kernel[grid_w](
                a_ptr=grad_y_t_int8, b_ptr=x_int8, c_ptr=grad_weight,
                a_scale_ptr=grad_y_t_scale, b_scale_ptr=dummy_b_scale, bias_ptr=grad_y_t_int8,
                M=N, N=K, K=M,
                stride_am=grad_y_t_int8.stride(0), stride_ak=grad_y_t_int8.stride(1),
                stride_bk=x_int8.stride(0), stride_bn=x_int8.stride(1),
                stride_cm=grad_weight.stride(0), stride_cn=grad_weight.stride(1),
                HAS_BIAS=False
            )

        if ctx.needs_input_grad[3] and ctx.has_bias:
            grad_bias = grad_output_2d.sum(dim=0)

        return grad_x, grad_weight, grad_weight_scale, grad_bias, None, None

# =============================================================================
# Python Wrappers (Now wrapping Autograd)
# =============================================================================

def triton_int8_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale, bias=None, compute_dtype=torch.float16):
    """
    Fused pipeline for W8A8 Linear Layer. (tensorwise weight scale)
    *Includes a fused row-wise backward pass.*
    """
    return TritonInt8LinearFunction.apply(x, weight, weight_scale, bias, compute_dtype, False)

def triton_int8_linear_per_row(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, bias=None, compute_dtype=torch.float16):
    """
    Fused pipeline for W8A8 Linear Layer with per-row weight quantization.
    *Includes a fused row-wise backward pass.*
    """
    return TritonInt8LinearFunction.apply(x, weight, weight_scale, bias, compute_dtype, True)