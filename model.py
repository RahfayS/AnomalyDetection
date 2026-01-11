import torch
import torch.nn as nn
import numpy as np

# ==== Encoder Block ====
class EncoderBlock(nn.Module):
  """Convolutional Block for Encoder"""
  def __init__(self,in_c,out_c):
    super().__init__()

    # Sample input: (b,1,256,256)

    # --- Layer 1 ---
    self.layer1 = nn.Sequential(
      nn.Conv2d(in_c,out_c,kernel_size=3,stride=2,padding=1), # (b,out_c,128,128)
      nn.BatchNorm2d(out_c),
      nn.LeakyReLU(0.2,inplace=True)
    )

    # --- Layer 2 ---
    self.layer2 = nn.Sequential(
      nn.Conv2d(out_c,out_c,kernel_size=3,stride=2,padding=1),  # (b,out_c,64,64)
      nn.BatchNorm2d(out_c),
      nn.LeakyReLU(0.2,inplace=True)
    )

    # --- Shortcut ---
    self.shortcut = nn.Sequential(
      nn.Conv2d(in_c, out_c, kernel_size=3, stride=4, padding=1), # (b,out_c,256,256) -> (b,out_c,64,64)
      nn.BatchNorm2d(out_c)
    )

    # --- Activation Function ---
    self.activation = nn.LeakyReLU(0.2, inplace=True)

  def forward(self,x):
    # x: (b,out_c,256,256)
    residual = self.shortcut(x) # Extract shortcut: (b,1,64,64)

    # pass input through conv block
    x = self.layer1(x) # x: (b,out_c,256,256) -> (b,out_c,128,128)
    x = self.layer2(x) # x: (b,out_c,128,128) -> (b,out_c,64,64)

    return self.activation(x + residual) # add residual to conv block output


# ==== Encoder ====
class Encoder(nn.Module):
  """ Takes some input X and downsamples to a compressed version """
  def __init__(self,channels,ch=256,z = 32):
    super().__init__()

    self.conv1 = nn.Conv2d(channels, ch, 3, 1, 1) # (b,256,256,256)

    # --- Pass through Convolutional Blocks ---
    self.block1 = EncoderBlock(ch,ch) # (b,ch,64,64)
    self.block2 = EncoderBlock(ch,ch*2) # (b,ch*2,16,16)
    self.block3 = EncoderBlock(ch*2,ch*4) # (b,ch*4,4,4)

    self.conv_mu = nn.Conv2d(ch * 4, out_channels= z, kernel_size=1,stride = 1)
    self.conv_logvar = nn.Conv2d(ch * 4, out_channels= z, kernel_size=1,stride = 1)

  def sample_latent(self,mu,logvar):
    std = torch.exp(0.5*logvar) # Compute the log variance
    epsilon = torch.randn_like(std) # Sample from the standard normal distribution N(0,1) of the same shape as the std

    return mu + epsilon * std # return sample

  def forward(self,x):
    x = self.conv1(x)
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)

    mu = self.conv_mu(x)
    logvar = self.conv_logvar(x)

    z = self.sample_latent(mu, logvar)


    return z,mu,logvar


# ==== Decoder Block ====
class DecoderBlock(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()

    self.block = nn.Sequential(
      nn.ConvTranspose2d(
        in_c, out_c,
        kernel_size=4,
        stride=2,
        padding=1
        ),
      nn.BatchNorm2d(out_c),
      nn.LeakyReLU(0.2, inplace=True)
  )

  def forward(self, x):
    return self.block(x)


class Decoder(nn.Module):
  def __init__(self,channels,ch=256,z=32):
    super().__init__()

    # Initial : # (b,1,4,4)

    self.block1 = DecoderBlock(z, 4 * ch) # 4 -> 8
    self.block2 = DecoderBlock(4 * ch, 2 * ch) # 8 -> 16
    self.block3 = DecoderBlock(2 * ch, ch)       # 16 -> 32
    self.block4 = DecoderBlock(ch, ch // 2)      # 32 -> 64
    self.block5 = DecoderBlock(ch // 2, ch // 4) # 64 -> 128
    self.block6 = DecoderBlock(ch // 4, ch // 8) # 128 -> 128

    self.out = nn.Sequential(
      nn.Conv2d(ch // 8, channels, kernel_size=3, padding=1),
      nn.Tanh()
    )

  def forward(self,z):


    x = self.block1(z)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)

    return self.out(x)


# --- Variational AutoEncoder ---
class VAE(nn.Module):
  def __init__(self,channel_in,ch = 256,z = 32):
    super().__init__()
    self.encoder = Encoder(channels=channel_in, ch=ch, z=z)
    self.decoder = Decoder(channels=channel_in, ch=ch, z=z)

  def forward(self, x):
    z, mu, logvar = self.encoder(x)

    # Only sample during training or when we want to generate new images
    # just use mu otherwise
    if self.training:
      decoding = self.decoder(z)
    else:
      decoding = self.decoder(mu)

    return decoding, mu, logvar