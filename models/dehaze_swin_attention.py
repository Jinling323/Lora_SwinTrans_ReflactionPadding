"""DehazeFormer-style spatial aggregation for torchvision Swin attention."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.swin_transformer import ShiftedWindowAttention


def _reflection_indices(length, pad_before, pad_after, device):
    """Return indices for reflection padding without PyTorch's pad-size limit."""
    if length <= 0:
        raise ValueError('Cannot pad an empty spatial dimension')
    if length == 1:
        return torch.zeros(
            length + pad_before + pad_after, dtype=torch.long, device=device
        )
    positions = torch.arange(
        -pad_before, length + pad_after, dtype=torch.long, device=device
    )
    period = 2 * (length - 1)
    positions = torch.remainder(positions, period)
    return torch.where(positions < length, positions, period - positions)


def reflection_pad_nchw(x, padding):
    """Reflection-pad NCHW tensors, including pads as large as the input."""
    left, right, top, bottom = padding
    height, width = x.shape[-2:]
    h_index = _reflection_indices(height, top, bottom, x.device)
    w_index = _reflection_indices(width, left, right, x.device)
    return x.index_select(-2, h_index).index_select(-1, w_index)


def _window_partition(x, window_size):
    batch, height, width, channels = x.shape
    window_h, window_w = window_size
    x = x.view(
        batch,
        height // window_h,
        window_h,
        width // window_w,
        window_w,
        channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).reshape(
        -1, window_h * window_w, channels
    )


def _window_reverse(windows, window_size, batch, height, width):
    window_h, window_w = window_size
    channels = windows.shape[-1]
    x = windows.view(
        batch,
        height // window_h,
        width // window_w,
        window_h,
        window_w,
        channels,
    )
    return x.permute(0, 1, 3, 2, 4, 5).reshape(
        batch, height, width, channels
    )


class DehazeShiftedWindowAttention(nn.Module):
    """Swin attention with optional parallel DWConv and reflection windows.

    The DWConv consumes the complete, linearly projected V feature map. Only
    the attention branch is partitioned into windows, matching DehazeFormer.
    """

    def __init__(self, source, parallel_dwconv, reflection_attention):
        super().__init__()
        if not isinstance(source, ShiftedWindowAttention):
            raise TypeError('source must be torchvision ShiftedWindowAttention')
        if not (parallel_dwconv or reflection_attention):
            raise ValueError('At least one Dehaze attention option must be enabled')

        self.window_size = list(source.window_size)
        self.shift_size = list(source.shift_size)
        self.num_heads = source.num_heads
        self.attention_dropout = source.attention_dropout
        self.dropout = source.dropout
        self.parallel_dwconv_enabled = parallel_dwconv
        self.reflection_attention = reflection_attention

        # Keep the pretrained modules and parameter names intact. Q/V LoRA is
        # attached to this same packed qkv Linear after the replacement.
        self.qkv = source.qkv
        self.proj = source.proj
        self.relative_position_bias_table = source.relative_position_bias_table
        self.register_buffer(
            'relative_position_index', source.relative_position_index.clone()
        )

        if parallel_dwconv:
            dim = source.qkv.in_features
            self.parallel_dwconv = nn.Conv2d(
                dim, dim, kernel_size=5, padding=0, groups=dim, bias=True
            )
            # Preserve the pretrained attention function at initialization.
            nn.init.zeros_(self.parallel_dwconv.weight)
            nn.init.zeros_(self.parallel_dwconv.bias)

    def get_relative_position_bias(self):
        window_tokens = self.window_size[0] * self.window_size[1]
        bias = self.relative_position_bias_table[
            self.relative_position_index
        ].view(window_tokens, window_tokens, -1)
        return bias.permute(2, 0, 1).contiguous().unsqueeze(0)

    def _pad_for_attention(self, x):
        """Pad NHWC features and return padded data plus active shift."""
        _, height, width, _ = x.shape
        window_h, window_w = self.window_size
        shift_h, shift_w = self.shift_size
        pad_bottom = (window_h - height % window_h) % window_h
        pad_right = (window_w - width % window_w) % window_w

        if self.reflection_attention:
            if shift_h or shift_w:
                pad_top = shift_h
                pad_left = shift_w
                pad_bottom = (window_h - shift_h + pad_bottom) % window_h
                pad_right = (window_w - shift_w + pad_right) % window_w
            else:
                pad_top = 0
                pad_left = 0
            x = reflection_pad_nchw(
                x.permute(0, 3, 1, 2),
                (pad_left, pad_right, pad_top, pad_bottom),
            ).permute(0, 2, 3, 1).contiguous()
            return x, [shift_h, shift_w], (pad_top, pad_left)

        # Preserve torchvision's original zero-padding/cyclic-shift behavior
        # when only the parallel convolution is requested.
        x = F.pad(x, (0, 0, 0, pad_right, 0, pad_bottom))
        padded_height, padded_width = x.shape[1:3]
        active_shift = [shift_h, shift_w]
        if window_h >= padded_height:
            active_shift[0] = 0
        if window_w >= padded_width:
            active_shift[1] = 0
        if active_shift[0] or active_shift[1]:
            x = torch.roll(
                x, shifts=(-active_shift[0], -active_shift[1]), dims=(1, 2)
            )
        return x, active_shift, (0, 0)

    def _add_cyclic_attention_mask(
        self, attention, padded_height, padded_width, active_shift
    ):
        shift_h, shift_w = active_shift
        if self.reflection_attention or not (shift_h or shift_w):
            return attention

        window_h, window_w = self.window_size
        mask = attention.new_zeros((padded_height, padded_width))
        h_slices = (
            (0, -window_h),
            (-window_h, -shift_h),
            (-shift_h, None),
        )
        w_slices = (
            (0, -window_w),
            (-window_w, -shift_w),
            (-shift_w, None),
        )
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                mask[
                    h_slice[0]:h_slice[1], w_slice[0]:w_slice[1]
                ] = count
                count += 1

        num_windows = (
            padded_height // window_h * padded_width // window_w
        )
        mask = mask.view(
            padded_height // window_h,
            window_h,
            padded_width // window_w,
            window_w,
        )
        mask = mask.permute(0, 2, 1, 3).reshape(
            num_windows, window_h * window_w
        )
        mask = mask.unsqueeze(1) - mask.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
        attention = attention.view(
            -1,
            num_windows,
            self.num_heads,
            window_h * window_w,
            window_h * window_w,
        )
        attention = attention + mask.unsqueeze(0).unsqueeze(2)
        return attention.view(
            -1,
            self.num_heads,
            window_h * window_w,
            window_h * window_w,
        )

    def forward(self, x):
        batch, height, width, channels = x.shape

        # DehazeFormer projects Q/K/V on the complete feature map, then
        # reflection-pads the projected tensor. Pointwise projection commutes
        # with reflection, and this also lets the DWConv reuse the same V.
        if self.reflection_attention:
            projected_qkv = self.qkv(x)
            if self.parallel_dwconv_enabled:
                value_map = projected_qkv[..., 2 * channels:3 * channels]
            padded, active_shift, crop_origin = self._pad_for_attention(
                projected_qkv
            )
            windows = _window_partition(padded, self.window_size)
            qkv = windows.reshape(
                windows.shape[0],
                windows.shape[1],
                3,
                self.num_heads,
                channels // self.num_heads,
            ).permute(2, 0, 3, 1, 4)
        else:
            padded, active_shift, crop_origin = self._pad_for_attention(x)
            windows = _window_partition(padded, self.window_size)
            qkv = self.qkv(windows).reshape(
                windows.shape[0],
                windows.shape[1],
                3,
                self.num_heads,
                channels // self.num_heads,
            ).permute(2, 0, 3, 1, 4)
            if self.parallel_dwconv_enabled:
                dim = self.qkv.in_features
                value_map = F.linear(
                    x,
                    self.qkv.weight[2 * dim:3 * dim],
                    None if self.qkv.bias is None else self.qkv.bias[2 * dim:3 * dim],
                )

        if self.parallel_dwconv_enabled:
            conv_value = reflection_pad_nchw(
                value_map.permute(0, 3, 1, 2), (2, 2, 2, 2)
            )
            conv_out = self.parallel_dwconv(conv_value).permute(0, 2, 3, 1)

        padded_height, padded_width = padded.shape[1:3]
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = query * (channels // self.num_heads) ** -0.5
        attention = query.matmul(key.transpose(-2, -1))
        attention = attention + self.get_relative_position_bias()
        attention = self._add_cyclic_attention_mask(
            attention, padded_height, padded_width, active_shift
        )
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(
            attention, p=self.attention_dropout, training=self.training
        )
        attention_out = attention.matmul(value).transpose(1, 2).reshape(
            windows.shape[0], windows.shape[1], channels
        )

        attention_out = _window_reverse(
            attention_out,
            self.window_size,
            batch,
            padded_height,
            padded_width,
        )
        if self.reflection_attention:
            crop_top, crop_left = crop_origin
            attention_out = attention_out[
                :, crop_top:crop_top + height, crop_left:crop_left + width, :
            ]
        else:
            if active_shift[0] or active_shift[1]:
                attention_out = torch.roll(
                    attention_out,
                    shifts=(active_shift[0], active_shift[1]),
                    dims=(1, 2),
                )
            attention_out = attention_out[:, :height, :width, :]

        if self.parallel_dwconv_enabled:
            attention_out = attention_out + conv_out
        output = self.proj(attention_out)
        return F.dropout(output, p=self.dropout, training=self.training)


def add_dehaze_attention(features, parallel_dwconv, reflection_attention):
    """Replace all torchvision Swin attention modules in ``features``."""
    if not (parallel_dwconv or reflection_attention):
        return 0
    blocks = [
        module
        for module in features.modules()
        if hasattr(module, 'attn')
        and isinstance(module.attn, ShiftedWindowAttention)
    ]
    if not blocks:
        raise ValueError('No torchvision Swin attention blocks found')
    for block in blocks:
        block.attn = DehazeShiftedWindowAttention(
            block.attn,
            parallel_dwconv=parallel_dwconv,
            reflection_attention=reflection_attention,
        )
    return len(blocks)

