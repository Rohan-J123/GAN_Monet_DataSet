import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize models
gen_AB = Generator()  # Regular -> Monet
gen_BA = Generator()  # Monet -> Regular
disc_A = Discriminator()  # Discriminator for Regular images
disc_B = Discriminator()  # Discriminator for Monet images

# Loss functions and optimizers
criterion_gan = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

optimizer_g_AB = optim.Adam(gen_AB.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_g_BA = optim.Adam(gen_BA.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d_A = optim.Adam(disc_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d_B = optim.Adam(disc_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Transformations and Dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Replace with your image folders
dataset_regular = datasets.ImageFolder(root='path_to_regular_images', transform=transform)
dataset_monet = datasets.ImageFolder(root='path_to_monet_images', transform=transform)

loader_regular = DataLoader(dataset_regular, batch_size=16, shuffle=True)
loader_monet = DataLoader(dataset_monet, batch_size=16, shuffle=True)

# Training Loop (Simplified)
epochs = 100
for epoch in range(epochs):
    for real_A, real_B in zip(loader_regular, loader_monet):
        real_A = real_A[0].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        real_B = real_B[0].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Update Generators (AB and BA)
        fake_B = gen_AB(real_A)
        fake_A = gen_BA(real_B)

        loss_identity_AB = criterion_identity(gen_AB(real_B), real_B) * 5.0
        loss_identity_BA = criterion_identity(gen_BA(real_A), real_A) * 5.0

        loss_cycle_ABA = criterion_cycle(gen_BA(fake_B), real_A) * 10.0
        loss_cycle_BAB = criterion_cycle(gen_AB(fake_A), real_B) * 10.0

        loss_g_AB = criterion_gan(disc_B(fake_B), torch.ones_like(fake_B))
        loss_g_BA = criterion_gan(disc_A(fake_A), torch.ones_like(fake_A))

        total_loss_g = loss_g_AB + loss_g_BA + loss_cycle_ABA + loss_cycle_BAB + loss_identity_AB + loss_identity_BA

        optimizer_g_AB.zero_grad()
        optimizer_g_BA.zero_grad()
        total_loss_g.backward()
        optimizer_g_AB.step()
        optimizer_g_BA.step()

        # Update Discriminators (A and B)
        loss_d_A = criterion_gan(disc_A(real_A), torch.ones_like(real_A)) + criterion_gan(disc_A(fake_A.detach()), torch.zeros_like(fake_A))
        loss_d_B = criterion_gan(disc_B(real_B), torch.ones_like(real_B)) + criterion_gan(disc_B(fake_B.detach()), torch.zeros_like(fake_B))

        optimizer_d_A.zero_grad()
        optimizer_d_B.zero_grad()
        loss_d_A.backward()
        loss_d_B.backward()
        optimizer_d_A.step()
        optimizer_d_B.step()

    print(f"Epoch [{epoch+1}/{epochs}] completed.")

# Save the trained models
torch.save(gen_AB.state_dict(), "generator_AB.pth")
torch.save(gen_BA.state_dict(), "generator_BA.pth")
