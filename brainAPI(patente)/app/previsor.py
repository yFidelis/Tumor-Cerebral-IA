import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import cv2  # Ensure cv2 is imported here
from PIL import Image
import numpy as np

torch.manual_seed(123)

class classificador(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3)) 
        self.conv2 = nn.Conv2d(64, 64, (3,3))
        self.activation = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=14*14*64, out_features=256)
        self.linear2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 4)

    def forward(self, X):
        X = self.pool(self.bnorm(self.activation(self.conv1(X))))
        X = self.pool(self.bnorm(self.activation(self.conv2(X))))
        X = self.flatten(X)

        # Camadas densas
        X = self.activation(self.linear1(X))
        X = self.activation(self.linear2(X))
        
        # Saída
        X = self.output(X)

        return X

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

classificadorLoaded = classificador()
state_dict = torch.load('app/checkpoint.pth')
classificadorLoaded.load_state_dict(state_dict)
UPLOAD_FOLDER = 'app/static/uploads/'

def classificarImagem(nome):
    # Load and preprocess the image
    img = cv2.imread(nome)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Define the preprocessing transformation
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Preprocess the image
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # Set the model to evaluation mode
    classificadorLoaded.eval()

    # Forward pass
    output = classificadorLoaded(img_tensor)
    pred_class = output.argmax(dim=1).item()
    output = F.softmax(output, dim=1)
    output = output.detach().cpu().numpy()
    resultado = np.argmax(output[0])
    target_layer = classificadorLoaded.conv2
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    hook_forward = target_layer.register_forward_hook(forward_hook)
    hook_backward = target_layer.register_full_backward_hook(backward_hook)

    classificadorLoaded.zero_grad()
    output_tensor = classificadorLoaded(img_tensor)
    output_tensor[:, pred_class].backward()

    hook_forward.remove()
    hook_backward.remove()

    gradients = gradients[0]
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    activations = activations[0]
    heatmap = torch.sum(weights * activations, dim=1, keepdim=True)
    heatmap = F.relu(heatmap)
    heatmap = heatmap.squeeze().cpu().detach().numpy()
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Salvar heatmap sobreposto
    heatmap_visual = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_visual, 0.4, 0)

    filename = os.path.basename(nome)
    heatmap_filename = f"HM_{filename}"
    heatmap_path = os.path.join(UPLOAD_FOLDER, heatmap_filename)
    cv2.imwrite(heatmap_path, superimposed_img)

    doencas = ["Glioma", "Meningioma", "Sem Tumor", "Pituitaria"]
    return doencas[resultado], heatmap_filename