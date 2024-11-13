import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

engine = create_engine("sqlite:///stock_data.db")

with engine.connect() as conn:
        df = pd.read_sql_table('stock_data', conn)
        
# Define VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        mu = self.fc2_mu(h1)
        logvar = self.fc2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# Generate example data
np.random.seed(0)
data = df.pivot(index='date', columns='symbol', values='close').dropna()

# Model parameters
tickers = data.columns
input_dim = len(data)
hidden_dim = 64
latent_dim = 10
epochs = 10  # Reduced epochs for quick execution in this example

# Train VAE column by column and store results
reconstructed_data = pd.DataFrame(index=data.index, columns=data.columns)
synthetic_data = {ticker: [] for ticker in tickers}

for ticker in tickers:
    column_data = torch.tensor(data[ticker].values, dtype=torch.float32).unsqueeze(0)  # Shape [1, input_dim]
    model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(column_data)
        loss = vae_loss(recon_batch, column_data, mu, logvar)
        loss.backward()
        optimizer.step()
    
    # Reconstruction and synthetic data generation
    model.eval()
    with torch.no_grad():
        recon_column, _, _ = model(column_data)
        reconstructed_data[ticker] = recon_column.squeeze(0).numpy()
        
        # Generate synthetic samples
        for _ in range(3):  # Reduced to 3 samples per ticker for brevity
            z = torch.randn(1, latent_dim)
            synthetic_column = model.decode(z).squeeze(0).numpy()
            synthetic_data[ticker].append(synthetic_column)

# Display input data sample, reconstructed data sample, and synthetic data sample
input_sample = data.head(5)
reconstructed_sample = reconstructed_data.head(5)
synthetic_sample = {ticker: np.array(samples)[:, :5] for ticker, samples in synthetic_data.items()}

print(input_sample, reconstructed_sample, synthetic_sample)
