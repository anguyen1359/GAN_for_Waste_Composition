#Austin Nguyen
#%%
#Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.models import inception_v3
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#%%
#Preparing Data
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter("runs/sampleGAN")

df = pd.read_csv('waste_example.csv')

#Features
composition_names = ['Organic Waste', 'Paper', 'Cardboard',
                    'Plastic (PET)', 'Plastic (Other)', 
                    'Metals (Aluminum)', 'Metals (Iron and Steel)',
                    'Glass', 'Textiles', 'E-waste', 
                    'Rubber', 'Other']
composition_data = df[composition_names].values.astype(np.float32)


class WasteDataset(Dataset):
    def __init__(self, features):
        self.features = features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index]

dataset = WasteDataset(composition_data)

batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, 
                         shuffle=True)


#Discriminator and Generator

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, ):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x):
        return self.model(x)

#Creating a discriminator and generator object
input_size_dis = len(composition_names)
hidden_size = 64 #Adjusted as needed
output_size_dis = 1 #Output for discriminator b/c True/False

discriminator = Discriminator(input_size_dis, hidden_size, output_size_dis).to(device)
input_size_gen = 100
output_size_gen = input_size_dis

generator = Generator(input_size_gen, hidden_size, output_size_gen).to(device)

#Preparing for Training and creating training models
learning_rate = 0.005
num_epoch = 1000
criterion = nn.BCELoss()

#Creating Optimizers
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=learning_rate)

#TensorBoard Variables
n_total_steps = len(data_loader)
discriminator_losses = []
generator_losses = []

print(f'Learning Rate: {learning_rate}')
#Training Loop
for epoch in range(num_epoch):
    for i, batch_real in enumerate(data_loader):
        # Step 1: Train the Discriminator with real samples

        # Zero the gradients before backward pass
        discriminator.zero_grad()

        # Get real samples from the data loader
        batch_real = batch_real.to(device)

        # Define real labels (1s) for the real samples
        batch_real_labels = torch.ones(batch_real.size(0), output_size_dis).to(device)

        # Forward pass through the Discriminator with real samples
        output_real = discriminator(batch_real)

        # Calculate the Discriminator's loss with real samples
        loss_real = criterion(output_real, batch_real_labels)

        # Backpropagation and optimization for the Discriminator
        loss_real.backward()
        optimizer_discriminator.step()

        # Step 2: Train the Discriminator with fake samples (from the Generator)

        # Generate noise for fake samples with correct input size
        noise = torch.randn(batch_real.size(0), input_size_gen).to(device)

        # Generate fake samples using the Generator
        batch_fake = generator(noise)

        # Define fake labels (0s) for the fake samples
        batch_fake_labels = torch.zeros(batch_real.size(0), output_size_dis).to(device)

        # Forward pass through the Discriminator with fake samples
        output_fake = discriminator(batch_fake.detach())

        # Calculate the Discriminator's loss with fake samples
        loss_fake = criterion(output_fake, batch_fake_labels)

        # Backpropagation and optimization for the Discriminator
        loss_fake.backward()
        optimizer_discriminator.step()

        # Step 3: Train the Generator

        # Zero the gradients before backward pass
        generator.zero_grad()

        # Generate noise for fake samples
        noise = torch.randn(batch_real.size(0), input_size_gen).to(device)

        # Generate fake samples using the Generator
        batch_fake = generator(noise)
        
        # Define real labels (1s) for the fake samples to fool the Discriminator
        batch_fake_labels = torch.ones(batch_real.size(0), output_size_dis).to(device)

        # Forward pass through the Discriminator with fake samples generated by the Generator
        output_fake = discriminator(batch_fake)

        # Calculate the Generator's loss based on the output of the Discriminator
        loss_generator = criterion(output_fake, batch_fake_labels)

        # Backpropagation and optimization for the Generator
        loss_generator.backward()
        optimizer_generator.step()


        # Calculate and store the losses
        generator_losses.append(loss_generator.item())
        discriminator_losses.append(loss_real.item() + loss_fake.item())
        
        if (epoch) % 10 == 0:
            print(f"Epoch [{epoch}/{num_epoch}] Batch Loss - Real: {loss_real.item():.4f}, Fake: {loss_fake.item():.4f}, Generator: {loss_generator.item():.4f}")

        if (epoch) % 100 == 0:
            writer.add_scalar("Generator Loss", loss_generator.item(), epoch)
            writer.add_scalar("Discriminator Loss", loss_real.item() + loss_fake.item(), epoch)
        
            # Logging Generated Samples (once every 100 epochs)
            with torch.no_grad():
                generated_samples = generator(torch.randn(100, input_size_gen).to(device))
            generated_samples_np = generated_samples.cpu().numpy()
            generated_samples_images = (generated_samples_np.T * 100).astype(np.uint8)  # Scale to 0-100
            writer.add_images("generated_samples", generated_samples_images, dataformats='HW')

        

# Training finished
print("GAN training completed!")



#%%
#Evaluating the Model

# Set the generator to evaluation mode (important when using dropout or batch normalization)
generator.eval()

# Generate noise for fake samples
num_evaluation_samples = len(dataset)  # You can choose the number of fake samples you want to generate
noise = torch.randn(num_evaluation_samples, input_size_gen).to(device)

# Generate fake samples using the Generator
with torch.no_grad():  # To prevent computation of gradients
    generated_samples = generator(noise)


# Convert the generated_samples to a numpy array
generated_samples_np = generated_samples.cpu().numpy()

# Calculate the Mean Squared Error (MSE) between the generated samples and the real data
mse = mean_squared_error(composition_data, generated_samples_np)

#1.0353, 
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {math.sqrt(mse):.4f}")

#Plot losses
plt.figure()
plt.plot(range(len(generator_losses)), generator_losses, label='Generator Loss', color='blue')
plt.plot(range(len(discriminator_losses)), discriminator_losses, label='Discriminator Loss', color='red')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Generator and Discriminator Losses over Time')
plt.legend()
plt.show()

#Plotting individual values and compare between real and generated values
for i in range(len(composition_data)-2):
    try:
        ax = plt.axes(aspect="equal")
        plt.title(composition_names[i])
        plt.scatter(composition_data[:, i], 
                generated_samples_np[:, i])
        plt.xlabel("Real Values")
        plt.ylabel("Generated Values")
        lims = [0, 50]
        plt.xlim(lims)
        plt.ylim(lims)
        plt.show()
    except Exception as e:
        break

#PCA graphs
pca = PCA(n_components=2)
transformed_real = pca.fit_transform(composition_data)
transformed_generated = pca.fit_transform(generated_samples_np)

plt.title("PCA of Real Samples")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.scatter(transformed_real[:, 0], transformed_real[:, 1])
plt.show()

plt.title("PCA of Generated Samples")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.scatter(transformed_generated[:, 0], transformed_generated[:, 1])
plt.show()

#Kmean of PCA
testing_np = np.vstack((composition_data, generated_samples_np))
transformed_testing = pca.fit_transform(testing_np)

kmeans = KMeans(n_clusters=2).fit(transformed_testing)
cluster_label = kmeans.labels_


plt.title("PCA of Testing Data (Generated and Real) with Separate Colors")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.scatter(transformed_testing[:1000, 0], transformed_testing[:1000, 1], label="Real Samples", color='blue')
plt.scatter(transformed_testing[1000:, 0], transformed_testing[1000:, 1], label="Generated Samples", color='red')
plt.legend()
plt.show()

plt.title("PCA of Testing Data (Generated and Real) with KMeans")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.scatter(transformed_testing[:, 0], transformed_testing[:, 1], c=cluster_label, cmap="rainbow")
plt.show()

# Plotting Histograms for Real and Generated Samples
def plot_histograms(data, title):
    plt.figure()
    plt.hist(data, bins=10, alpha=0.7, rwidth=0.85)
    plt.title(title)
    plt.xlabel("Composition Values")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    return plt

# Plot histograms for real and generated samples
plt_real = plot_histograms(composition_data.T, "Histogram of Real Waste Composition")
plt_generated = plot_histograms(generated_samples_np.T, "Histogram of Generated Waste Composition")

# Add histograms to TensorBoard writer
#writer.add_figure("Histogram/Real", plt_real)
#writer.add_figure("Histogram/Generated", plt_generated)

#%%
#Create a flow diagram of models 
writer.add_graph(generator, noise)
#%%
writer.add_graph(discriminator, generator(noise))

print("Evaluation Complete")



#%%
#Sample Generated Data
for i in range(10):
    print(generated_samples[i])

#%%
#Exporting Data
samples_wanted = 10000 #vary to output # of samples
noise = torch.randn(samples_wanted, input_size_gen).to(device)

with torch.no_grad():
    generated_samples = generator(noise)

export_array = generated_samples.numpy()
export_file = pd.DataFrame(export_array)
#export_file.to_csv("GAN_Generated_Samples.csv", header=composition_names, index=False)

# %%
#Saving Model

torch.save(generator.state_dict(), "SampleGAN_Trained_Generator.pth")
torch.save(discriminator.state_dict(), "SampleGAN_Trained_Discriminator.pth")