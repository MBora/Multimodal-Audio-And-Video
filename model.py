TRAIN_LABELS = {"LionTrain": 0, "CatTrain": 1}
# VALIDATION_LABELS = {"car2":0, "bike2": 1}
TEST_LABELS = {"LionTest":0, "CatTest": 1}

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from spectrogram_loader import SpectrogramDataLoader
from frame_loader import FrameDataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--spec_dir', '-s', help="Spectrograms directory", type=str)
parser.add_argument('--frame_dir', '-f', help="Frames directory", type=str)
args = parser.parse_args()
spec_dir = args.spec_dir
frame_dir = args.frame_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#transformations
import torchvision.transforms as transforms
transform = transforms.Compose([
    # transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

#Load data
train_loader_audio = SpectrogramDataLoader(root_dir=spec_dir, label_dict=TRAIN_LABELS, transform=transform)
train_loader_video = FrameDataLoader(root_dir=frame_dir, label_dict=TRAIN_LABELS, transform=transform)

# validation_loader_audio = SpectrogramDataLoader(root_dir=spec_dir, label_dict=VALIDATION_LABELS)
# validation_loader_video = FrameDataLoader(root_dir=frame_dir, label_dict=VALIDATION_LABELS)

test_loader_audio = SpectrogramDataLoader(root_dir=spec_dir, label_dict=TEST_LABELS)
test_loader_video = FrameDataLoader(root_dir=frame_dir, label_dict=TEST_LABELS)

#Use 3D CNN model for audio spectrogram classification
import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
num_epochs = 1
learning_rate = 0.001

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#multimodal 3D CNN model
class Conv3D(nn.Module):
    def __init__(self, num_classes=2):
        super(Conv3D, self).__init__()
        self.layer1_audio = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(5,5,5), padding=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=2),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=2)
        )
        self.layer1_video = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(5,5,5), padding=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=2),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(5,5,5), padding=2),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2,2,2), stride=2),
            nn.MaxPool3d(kernel_size=(3,3,3), stride=3)
        )
        self.fc = nn.Linear(5120, num_classes)
    
    def forward(self, audio, video):
        out_audio = self.layer1_audio(audio)
        out_video = self.layer1_video(video)
        out = torch.cat((out_audio, out_video), dim=1)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

#train the model

model = Conv3D()
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(zip(train_loader_audio, train_loader_video)):
        # Get the inputs and labels for both modalities
        inputs_audio, labels_audio = data[0]
        inputs_video, labels_video = data[1]
        # input(f"Audio = {len(inputs_audio)}, Video = {len(inputs_video)}")
        input(f"Audio = {inputs_audio.shape}, Video = {inputs_video.shape}")

        # Concatenate the feature maps from both modalities
        # inputs = torch.cat([inputs_audio, inputs_video], dim=4).to(device)
        labels = labels_audio.to(device)
        # input(f"Shape = {inputs.shape}")
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs_audio, inputs_video)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # Print every 10 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader_audio)}], Loss: {running_loss/10:.4f}")
            running_loss = 0.0
