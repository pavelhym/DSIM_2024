import torch
import torch.nn as nn
import streamlit as st



class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(

          nn.ConvTranspose2d(nz, 512, kernel_size=4, stride=1, padding=0, bias=False),
          nn.BatchNorm2d(512),
          nn.ReLU(True),

          nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(256),
          nn.ReLU(True),

          nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(128),
          nn.ReLU(True),

          nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(64),
          nn.ReLU(True),

          nn.ConvTranspose2d(64, 3, kernel_size=4,stride=2, padding=1, bias=False),
          nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


import torchvision.transforms as transforms
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

mean = torch.tensor([0.5, 0.5, 0.5])
std = torch.tensor([0.5, 0.5, 0.5])
unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())


generator = Generator(100)
generator.load_state_dict(torch.load('gan/generator.pth', map_location=torch.device('cpu')))


def predict(n):
    noise = torch.randn(n, 100, 1, 1)
    generated_img = generator(noise)
    return generated_img


def show_gan_imgs(n):
    generated_img = predict(n)
    unnormalized = unnormalize(generated_img).cpu().detach().numpy()
    fig = plt.figure()
    if (n > 5) & (n%2 == 0):
        columns = int(n/2)
        rows = 2
    else:
        columns = n
        rows = 1
    for i in range(1, columns*rows +1):
    
      fig.add_subplot(rows, columns, i)
      plt.imshow(np.transpose(unnormalized[i-1], (1, 2, 0)))
      plt.gca().xaxis.set_visible(False)
      plt.gca().yaxis.set_ticks([])

    st.pyplot(fig)
