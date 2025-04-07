class SimpleAddFusion(nn.Layer):
    """
    A simplified feature fusion module which:
      1) Convolves the low-level feature x.
      2) Upsamples the high-level feature y to match x's spatial size.
      3) Element-wise adds x and y.
      4) Convolves the result to produce final output.

    Args:
        x_ch (int): The channel of x tensor (low-level feature).
        y_ch (int): The channel of y tensor (high-level feature).
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the convolution for x tensor. Default: 3.
        resize_mode (str, optional): The upsampling mode for y tensor. Default: 'bilinear'.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()
        # Convolution applied to the low-level feature x
        self.conv_x = layers.ConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias_attr=False
        )
        # Convolution applied after x + y
        self.conv_out = layers.ConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias_attr=False
        )
        self.resize_mode = resize_mode

    def check(self, x, y):
        """
        Check shape dimensions to ensure x is not smaller
        than y in spatial resolution.
        """
        assert x.ndim == 4 and y.ndim == 4, "Tensors must be 4D (N, C, H, W)."
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w, \
            "Low-level feature (x) must have >= spatial size than high-level feature (y)."

    def prepare(self, x, y):
        """
        Apply the 'prepare_x' and 'prepare_y' steps to
        convolve x and upsample y to match x's shape.
        """
        x = self.prepare_x(x)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x):
        """
        Convolve the low-level feature x to match y's channel dimension.
        """
        x = self.conv_x(x)
        return x

    def prepare_y(self, x, y):
        """
        Upsample the high-level feature y to the same height and width as x.
        """
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        return y_up

    def fuse(self, x, y):
        """
        Simple fusion by element-wise addition, followed by a convolution.
        """
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, x, y):
        """
        Perform shape checking, feature preparation, and fusion.

        Args:
            x (Tensor): The low-level feature (N, x_ch, Hx, Wx).
            y (Tensor): The high-level feature (N, y_ch, Hy, Wy).

        Returns:
            Tensor: The fused output feature of shape (N, out_ch, Hx, Wx).
        """
        self.check(x, y)
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out

