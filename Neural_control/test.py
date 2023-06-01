import torch
from torchvision.models import alexnet

# Create a random tensor with batch size 1 and 3 channels, 224x224 as an example input
input = torch.randn(1, 3, 224, 224)

# Get the pretrained AlexNet model
model = alexnet(pretrained=True)

# Pass the input through each layer and print the output shape
x = input
for layer in model.features:
    x = layer(x)
    print(f'After layer: {layer}, output shape is: {x.shape}')
