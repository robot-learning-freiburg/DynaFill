
def _get_same_conv_padding(input_size, kernel_size, stride, dilation=1):
    if input_size % stride == 0:
        total_padding = dilation * (kernel_size - 1) + 1 - stride
    else:
        total_padding = dilation * (kernel_size - 1) + 1 - (input_size % stride)

    total_padding = max(0, total_padding)

    start = total_padding // 2
    end = total_padding - start
    return start, end

