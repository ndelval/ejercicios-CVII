"""
Network layers for PixelRNN - Incomplete version.

This module contains the core building blocks for PixelRNN:
- MaskedConv2d: Convolutional layer with causal masking
- RowLSTM: LSTM that processes image rows with masked convolutions
- DiagonalBiLSTM: Bidirectional LSTM that processes image diagonally

Students need to implement the masking logic and LSTM forward passes.
"""

import torch
from torch import nn


class MaskedConv2d(nn.Module):
    """2D Convolution with causal masking for autoregressive models.

    This layer implements masked convolutions where the mask ensures that
    the prediction for a pixel only depends on previously generated pixels
    (pixels above and to the left in raster scan order).

    There are two types of masks:
    - Type 'A': Excludes the center pixel (used in first layer)
    - Type 'B': Includes the center pixel (used in subsequent layers)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        mask_type: str = "B",
        stride: int = 1,
        padding: int | str = "same",
    ) -> None:
        """Initialize masked convolution layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel (int or tuple)
            mask_type: Type of mask - 'A' (excludes center) or 'B' (includes center)
            stride: Stride of the convolution
            padding: Padding mode ('same' keeps spatial dimensions)
        """
        super().__init__()
        self.mask_type = mask_type

        # [ ] Initialize the Conv2d layer with appropriate parameters
        # Hint: Use nn.Conv2d with the given parameters
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        if isinstance(kernel_size, tuple):
            kernel_h, kernel_w = kernel_size
        else:
            kernel_h = kernel_size
            kernel_w = kernel_size
        self.mask = torch.ones((in_channels, out_channels, kernel_h, kernel_w))
        center_h = kernel_h // 2
        center_w = kernel_w // 2
        self.mask[..., center_h + 1 :, :] = 0
        self.mask[..., center_h, center_w + 1 :] = 0
        if mask_type == "A":
            self.mask[..., center_h, center_w] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with masked convolution.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor after masked convolution
        """
        self.conv.weight.data *= self.mask
        return self.conv(x)


class RowLSTM(nn.Module):
    """Row LSTM layer for PixelRNN.

    Processes each row of the image using an LSTM, where each position
    receives input from the current pixel and hidden state from the
    previous pixel in the row. Uses masked convolutions to incorporate
    context from above.

    The Row LSTM processes the image row by row, left to right,
    maintaining a hidden state that flows horizontally.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
    ) -> None:
        """Initialize Row LSTM layer.

        Args:
            in_channels: Number of input channels
            hidden_channels: Number of hidden channels in LSTM
        """
        super().__init__()
        self.hidden_channels = hidden_channels

        # Convolución masked para preprocesar la entrada (contexto de arriba/izquierda)
        self.input_conv = MaskedConv2d(
            in_channels,
            hidden_channels,
            kernel_size=(1, 3),
            mask_type="B",
        )

        # LSTM para procesar cada fila
        self.lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Row LSTM.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Hidden states tensor of shape [B, hidden_channels, H, W]
        """
        batch_size, _, height, width = x.shape

        # Preprocesar con masked conv
        # Shape: [B, hidden_channels, H, W]
        x = self.input_conv(x)

        # Reorganizar para LSTM: procesar cada fila como una secuencia
        # [B, hidden_channels, H, W] -> [B * H, W, hidden_channels]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, hidden_channels]
        x = x.reshape(batch_size * height, width, self.hidden_channels)

        # Pasar por LSTM
        output, _ = self.lstm(x)  # [B * H, W, hidden_channels]

        # Reorganizar de vuelta
        # [B * H, W, hidden_channels] -> [B, hidden_channels, H, W]
        output = output.reshape(batch_size, height, width, self.hidden_channels)
        output = output.permute(0, 3, 1, 2)  # [B, hidden_channels, H, W]

        return output


class DiagonalBiLSTM(nn.Module):
    """Diagonal Bidirectional LSTM for PixelRNN.

    This layer processes the image along diagonals, allowing information
    to flow from top-left to bottom-right and from top-right to bottom-left.
    This enables each pixel to have context from a larger receptive field
    compared to Row LSTM.

    The diagonal processing is achieved by skewing the feature map,
    processing with a column LSTM, and then unskewing.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
    ) -> None:
        """Initialize Diagonal BiLSTM layer.

        Args:
            in_channels: Number of input channels
            hidden_channels: Number of hidden channels in each direction
        """
        super().__init__()
        self.hidden_channels = hidden_channels

        self.input_conv = MaskedConv2d(
            in_channels,
            hidden_channels,
            kernel_size=(1, 3),
            mask_type="B",
        )

        # LSTM para procesar cada fila
        self.row_lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            batch_first=True,
        )
        self.col_lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            batch_first=True,
        )

    def _skew(self, x: torch.Tensor) -> torch.Tensor:
        """Skew the input tensor for diagonal processing.

        Shifts each row by its index to align diagonals into columns.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Skewed tensor of shape [B, C, H, W + H - 1]
        """
        B, C, H, W = x.shape

        # Nuevo ancho después del skew
        new_width = W + H - 1

        # Crear tensor de salida con ceros
        skewed = torch.zeros(B, C, H, new_width, device=x.device, dtype=x.dtype)

        # Para cada fila i, colocar los datos empezando en posición i
        for i in range(H):
            skewed[:, :, i, i : i + W] = x[:, :, i, :]

        return skewed

    def _unskew(self, x: torch.Tensor, original_width: int) -> torch.Tensor:
        """Unskew the tensor back to original shape.

        Reverses the skewing operation to restore original spatial layout.

        Args:
            x: Skewed tensor of shape [B, C, H, W + H - 1]
            original_width: Original width before skewing

        Returns:
            Unskewed tensor of shape [B, C, H, original_width]
        """
        B, C, H, _ = x.shape

        # Crear tensor de salida
        unskewed = torch.zeros(B, C, H, original_width, device=x.device, dtype=x.dtype)

        # Para cada fila i, extraer los datos desde posición i
        for i in range(H):
            unskewed[:, :, i, :] = x[:, :, i, i : i + original_width]

        return unskewed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Diagonal BiLSTM.

        Procesa la imagen con dos LSTMs:
        - row_lstm: contexto de la izquierda (procesa filas)
        - col_lstm: contexto de arriba (procesa columnas)

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, 2*hidden_channels, H, W]
        """
        # No uso el skew por que la diagonal lstm que hay en los apuntes creo que es distinta, e implemento esa
        batch_size, _, height, width = x.shape

        # Preprocesar con masked conv
        x = self.input_conv(x)  # [B, hidden_channels, H, W]

        # === Row LSTM (contexto izquierda) ===
        # [B, hidden_channels, H, W] -> [B*H, W, hidden_channels]
        x_rows = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, hidden_channels]
        x_rows = x_rows.view(batch_size * height, width, self.hidden_channels)

        row_out, _ = self.row_lstm(x_rows)  # [B*H, W, hidden_channels]

        # View de vuelta: [B*H, W, hidden_channels] -> [B, hidden_channels, H, W]
        row_out = row_out.view(batch_size, height, width, self.hidden_channels)
        row_out = row_out.permute(0, 3, 1, 2)  # [B, hidden_channels, H, W]

        # === Col LSTM (contexto arriba) ===
        # [B, hidden_channels, H, W] -> [B*W, H, hidden_channels]
        x_cols = x.permute(0, 3, 2, 1).contiguous()  # [B, W, H, hidden_channels]
        x_cols = x_cols.view(batch_size * width, height, self.hidden_channels)

        col_out, _ = self.col_lstm(x_cols)  # [B*W, H, hidden_channels]

        # View de vuelta: [B*W, H, hidden_channels] -> [B, hidden_channels, H, W]
        col_out = col_out.view(batch_size, width, height, self.hidden_channels)
        col_out = col_out.permute(0, 3, 2, 1)  # [B, hidden_channels, H, W]

        # Concatenar ambos contextos
        return torch.cat([row_out, col_out], dim=1)  # [B, 2*hidden_channels, H, W]
