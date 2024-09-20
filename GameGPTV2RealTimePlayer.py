import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pygame

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

value_to_int_mapping = {'Q': 0, 'L': 1, 'N': 2, 'R': 3}
int_to_value_mapping = {0: 'Q', 1: 'L', 2: 'N', 3: 'R'}

class ImageToVector(nn.Module):
    def __init__(self, input_channels, image_size, conv_size, d_model, dropout):  # Change output_size to 256
        super(ImageToVector, self).__init__()

        # First Convolutional block with ReLU and MaxPooling
        self.conv1 = nn.Conv2d(input_channels, conv_size, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolutional block with ReLU and MaxPooling
        self.conv2 = nn.Conv2d(conv_size, conv_size * 2, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        # Adjusting the input size of the fully connected layer to match new dimensions
        self.fc = nn.Linear((image_size // 4) * (image_size // 4) * conv_size * 2, d_model) 
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Second block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = self.relu(self.fc(x))
        x = self.dropout(x)

        return x


# Input Vector Positional Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):  # Dynamically set d_model
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_seq_len, d_model).to(device)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]


# Self Attention Layer
class Attention(nn.Module):
    def __init__(self, d_model, n_head, seq_len, dropout):
        super(Attention, self).__init__()
        
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len)))
        self.d_model = d_model
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.shape


        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        scale = C ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_scores = attn_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.dropout(self.linear(attn_output))  # Apply final linear transformation and dropout
        
        return self.norm(output + x)  # Apply residual connection and normalization

class StackedAttention(nn.Module):
    def __init__(self, d_model, n_head, seq_len, num_layers, dropout):
        super(StackedAttention, self).__init__()

        self.attention_layers = nn.ModuleList(
            [Attention(d_model, n_head, seq_len, dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        for attn_layer in self.attention_layers:
            x = attn_layer(x)  # Apply each attention layer sequentially
        return x

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_head, seq_len_q, seq_len_k, dropout):
        super(CrossAttention, self).__init__()

        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        
        self.c_attn_q = nn.Linear(d_model, d_model)
        self.c_attn_k = nn.Linear(d_model, d_model)
        self.c_attn_v = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Attention mask
        self.register_buffer('mask', torch.tril(torch.ones(seq_len_q, seq_len_k)))
        self.d_model = d_model
        self.n_head = n_head
        
    def forward(self, queries, keys, values):
        B, T_q, C = queries.shape
        _, T_k, _ = keys.shape
        
        # Apply linear projections separately
        q = self.c_attn_q(queries)  # (B, T_q, C)
        k = self.c_attn_k(keys)     # (B, T_k, C)
        v = self.c_attn_v(values)   # (B, T_k, C)
        
        # Reshape for multi-head attention
        q = q.view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T_q, C // n_head)
        k = k.view(B, T_k, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T_k, C // n_head)
        v = v.view(B, T_k, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T_k, C // n_head)
        
        # Scaled dot-product attention
        scale = C ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_head, T_q, T_k)
        
        # Apply attention mask
        attn_scores = attn_scores.masked_fill(self.mask[:T_q, :T_k] == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, n_head, T_q, T_k)
        attn_output = torch.matmul(attn_weights, v)  # (B, n_head, T_q, C // n_head)
        
        # Concatenate heads and apply final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, C)  # (B, T_q, C)
        output = self.out_linear(attn_output)  # (B, T_q, C)
        
        # Apply dropout and residual connection
        output = self.dropout(output)
        return self.norm(output + queries)  # Residual connection and normalization


# Basic Vector To Image decoder

class VectorToImage(nn.Module):
    def __init__(self, d_model, output_channels, image_size, conv_size, dropout):
        super(VectorToImage, self).__init__()
        self.output_channels = output_channels
        self.image_size = image_size
        self.conv_size = conv_size

        # Calculate the size of the feature map after the fully connected layer
        self.fc_out_size = conv_size * 8 * (image_size // 16) * (image_size // 16)

        # Linear layer to expand the feature vector to the size of the desired image
        self.fc = nn.Linear(d_model, self.fc_out_size)

        # Deconvolution layers
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(conv_size * 8, conv_size * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(conv_size * 4),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(conv_size * 4, conv_size * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(conv_size * 2),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(conv_size * 2, conv_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(conv_size),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(conv_size, output_channels, kernel_size=4, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # x has shape [BT, C] where BT = batch size * timesteps

        # Pass through the fully connected layer
        x = self.fc(x)
        
        # Reshape to match the dimensions for the first deconvolution layer
        x = x.view(-1, self.conv_size * 8, self.image_size // 16, self.image_size // 16)
        
        # Pass through the deconvolution layers
        x = self.deconv1(x)  # Upsample to (BT, conv_size * 4, H // 8, W // 8)
        x = self.deconv2(x)  # Upsample to (BT, conv_size * 2, H // 4, W // 4)
        x = self.deconv3(x)  # Upsample to (BT, conv_size, H // 2, W // 2)
        x = self.deconv4(x)  # Final output shape (BT, output_channels, H, W)
        x = self.dropout(x)
        x = F.sigmoid(x)
 
        return x


# Final model

class GameModel(nn.Module):
    def __init__(self, d_model, image_size, channels, num_input_tokens, seq_len, n_head, num_a_layers, ca_n_head,conv_size, dropout):
        super(GameModel, self).__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.n_head = n_head

        # Embedding layer for input tokens
        self.input_token_embedding = nn.Embedding(num_input_tokens, d_model)
        
        # Image to vector transformation
        self.image_to_vector = ImageToVector(input_channels=channels, image_size=image_size, conv_size=conv_size, d_model=d_model, dropout=dropout)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len=seq_len)
        
        # Stacked attention layers
        self.stacked_attention = StackedAttention(d_model, n_head, seq_len, num_a_layers, dropout)

        # Cross-attention layer
        self.cross_attention = CrossAttention(d_model, ca_n_head, seq_len_q=seq_len, seq_len_k=seq_len, dropout=dropout)
        
        # Output layer for image generation
        self.vector_to_image = VectorToImage(d_model=d_model, output_channels=channels, image_size=image_size, conv_size=conv_size, dropout=dropout)

    def forward(self, k, x):
        # Extract batch size and number of timesteps
        B, T, _, _, _ = x.shape

        # Process all timesteps in parallel
        # Convert images to vectors
        image_vectors = self.image_to_vector(x.view(-1, *x.shape[2:]))  # Flatten batch and timesteps
        image_vectors = image_vectors.view(B, T, -1)  # Reshape to (B, T, d_model)

        # Add positional encoding
        image_vectors = self.positional_encoding(image_vectors)
        
        # Embed input tokens
        k_embeddings = self.input_token_embedding(k)  # (B, T, d_model)

        # Apply stacked self-attention
        y = self.stacked_attention(image_vectors)
        
        # Apply cross-attention between token embeddings and self-attention output
        y = self.cross_attention(queries=y, keys=k_embeddings, values=k_embeddings)  # Cross-attention
        
        # Add residual connection and apply layer normalization
        y = image_vectors + y
        
        # Generate images from vectors
        generated_images = self.vector_to_image(y.view(-1, self.d_model))  # Flatten timesteps
        generated_images = generated_images.view(B, T, *generated_images.shape[1:])  # Reshape to (B, T, C, H, W)

        return generated_images

seq_len = 5
d_model = 1024
image_size = 128
conv_size = 64
n_head = 16
num_a_layers = 6
ca_n_head = 1
num_input_tokens = 4
dropout = 0.3
channels = 3


# Initialize pygame
pygame.init()

# Constants
IMAGE_SIZE = (128, 128)
SCREEN_SIZE = (512, 512)
FPS = 60

# Setup screen
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Real-Time Gameplay Renderer")

# Load model (assuming it's already trained and ready)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GameModel(d_model=d_model, image_size=image_size, channels=channels, num_input_tokens=num_input_tokens, seq_len=seq_len, n_head=n_head, num_a_layers=num_a_layers, ca_n_head=ca_n_head, conv_size=conv_size, dropout=dropout)
model.to(device)
model_path = f"../model_final_xl_v7.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

# Image transformation
data_transform_p = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# Game environment
class PlayDataset:
    def __init__(self):
        self.current_frame = Image.open("../content/frames_7/frame_0.png")
        self.frames = [self.current_frame] * 5
        self.moves = [2, 2, 2, 2, 2]  # Example initial moves

    def update(self, move, new_frame):
        self.frames.pop(0)
        self.frames.append(new_frame)
        self.moves.pop(0)
        self.moves.append(value_to_int_mapping[move])

    def get_input(self):
        frames = [data_transform_p(frame) for frame in self.frames]
        frames_tensor = torch.stack(frames).unsqueeze(0).to(device)
        moves_tensor = torch.tensor(self.moves).unsqueeze(0).to(device)
        return moves_tensor, frames_tensor

play_dataset = PlayDataset()

# Game loop
running = True
clock = pygame.time.Clock()

key_mapping = {
    pygame.K_LEFT: 'L',
    pygame.K_RIGHT: 'R',
    pygame.K_UP: 'L',  # Add if needed for up arrow
    pygame.K_DOWN: 'R'  # Add if needed for down arrow
}

font = pygame.font.SysFont('Arial', 24)

while running:
    
    for event in pygame.event.get():
        move = "N"
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in key_mapping:
                move = key_mapping[event.key] 
           
    # Get input for model
    moves, frames = play_dataset.get_input()

    # Forward pass
    with torch.no_grad():
        output = model(moves, frames)

    # Process output frame
    predicted_frames = output.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
    predicted_frame = (predicted_frames[-1] * 255).astype('uint8')  # Get the last frame predicted
    image_pil = Image.fromarray(predicted_frame)

    # Update the play dataset with the new frame
    play_dataset.update(move, image_pil)  # Simulating a movement, replace "R" as needed

    # Convert to a surface and display
    new_size = (512, 512)
    image_resized = image_pil.resize(new_size, Image.NEAREST)
    mode = 'RGB'
    size = image_resized.size
    data = image_resized.tobytes()
    image_surface = pygame.image.fromstring(data, size, mode)

    # Render to screen
    screen.fill((0, 0, 0))  # Clear screen
    screen.blit(image_surface, (0, 0))  # Draw the new frame

    # **Display FPS on the screen** 
    fps = clock.get_fps()
    fps_text = font.render(f"FPS: {fps:.2f}", True, (255, 255, 255))
    screen.blit(fps_text, (10, 10))  # Render FPS in the top-left corner


    pygame.display.flip()

    clock.tick(FPS)  # Maintain frame rate

pygame.quit()