import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .feedforwardNN import FeedforwardNN

class CVAEIntegrated(nn.Module):
    def __init__(self,
                 design_dim,      # 4 (A parameters)
                 spectrum_dim,    #  5000 (B)
                 cond_dim,        #  2 (C)
                 latent_dim,
                 hidden_dims_encoder,
                 hidden_dims_decoder,
                 hidden_dims_fnn=None,
                 dropoutFNN=None,
                 hidden_dim_classifier=None,
                 dropoutCVAE=0.0,
                 beta=0.05,
                 gamma=None):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.cond_dim = cond_dim

        # Encoder
        self.encoder = Encoder(design_dim + spectrum_dim + cond_dim,
                               latent_dim,
                               hidden_dim_list=hidden_dims_encoder,
                               dropout=dropoutCVAE)
        # Generator
        self.generator = Decoder(latent_dim + cond_dim,
                                 design_dim,
                                 hidden_dim_list=hidden_dims_decoder,
                                 dropout=dropoutCVAE)

        # Classifier (optionnel)
        self.use_classifier = hidden_dim_classifier is not None and gamma is not None
        if self.use_classifier:
            self.latent_classifier = LatentClassifier(
                latent_dim, cond_dim, hidden_dim=hidden_dim_classifier, dropout=None
            )

        # Predictor: FNN
        if hidden_dims_fnn is None or dropoutFNN is None:
            # Charge le modèle FNN pré-entraîné
            checkpoint = torch.load("saved_models/ffn_big_trained.pth", map_location="cpu")
            self.predictor = FeedforwardNN(
                checkpoint["input_dim"],
                checkpoint["output_dim"],
                checkpoint["hidden_dim_list"],
                checkpoint["dropout"]
            )
            self.predictor.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Utilise le FNN défini par les arguments
            self.predictor = FeedforwardNN(
                design_dim,
                spectrum_dim,
                hidden_dims_fnn,
                dropoutFNN
            )

    def encode(self, A, B, C):
        x = torch.cat([A, B, C], dim=1)
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def generate_design(self, z, C):
        z_cond = torch.cat([z, C], dim=1)
        A_hat = self.generator(z_cond)
        return A_hat

    def predict_spectrum(self, A_hat):
        self.predictor.eval()  # Assurez-vous que le modèle est en mode évaluation
        if not isinstance(A_hat, torch.Tensor):
            A_hat = torch.tensor(A_hat, dtype=torch.float32)
        else:
            A_hat = A_hat.clone().detach().float()
        B_pred = self.predictor(A_hat)
        return B_pred

    def forward(self, A, B, C):
        mu, logvar = self.encode(A, B, C)
        z = self.reparameterize(mu, logvar)
        A_hat = self.generate_design(z, C)
        B_pred = self.predict_spectrum(A_hat)
        if self.use_classifier:
            C_pred = self.latent_classifier(mu)
        else:
            C_pred = None
        return A_hat, B_pred, mu, logvar, C_pred

    def decode(self, z, C):
        A_hat = self.generate_design(z, C)
        B_pred = self.predict_spectrum(A_hat)
        return A_hat, B_pred

    def loss(self, A_hat, A, B_pred, B, mu, logvar, C_true, C_pred):
        loss_rec = F.mse_loss(A_hat, A, reduction='mean')
        loss_pred = F.mse_loss(B_pred, B, reduction='mean')
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        if self.use_classifier and C_pred is not None and C_true is not None:
            loss_cls = F.mse_loss(C_pred, C_true, reduction='mean')
            total_loss = loss_rec + loss_pred + self.beta * kl + self.gamma * loss_cls
        else:
            loss_cls = torch.tensor(0.0, device=A_hat.device)
            total_loss = loss_rec + loss_pred + self.beta * kl
        return total_loss, loss_rec, loss_pred, kl
