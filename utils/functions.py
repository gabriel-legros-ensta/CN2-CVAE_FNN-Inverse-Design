import json
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from CVAE.cvae import CVAEIntegrated
import torch
import plotly.express as px
import umap
import streamlit as st
from scipy.signal import find_peaks, filtfilt
from scipy.signal.windows import gaussian
from torch.distributions import MultivariateNormal
import pandas as pd


def load_best_params(json_path="best_params.json"):
    with open(json_path, "r") as f:
        best_params = json.load(f)
    latent_dim = best_params["latent_dim"]
    hidden_dim_list = best_params["hidden_dim_list"]
    lr = best_params["lr"]
    dropout = best_params["dropout"]
    print("Best parameters found:")
    print("Latent dimension:", latent_dim)
    print("Hidden dimensions:", hidden_dim_list)
    print("Learning rate:", lr)
    print("Dropout rate:", dropout)
    return latent_dim, hidden_dim_list, lr, dropout

def pca(dataset, variance_threshold=0.99):
    """
    Calcule le nombre minimal de composantes principales expliquant variance_threshold de la variance totale.
    Affiche la forme concaténée et le nombre de composantes nécessaires.
    """
    X_all = dataset.tensors[0].cpu().numpy()
    Y_all = dataset.tensors[1].cpu().numpy()
    XY_data = np.concatenate([X_all, Y_all], axis=1)
    print("Shape of concatenated data (X, Y):", XY_data.shape)

    pca = PCA()
    pca.fit(XY_data)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    latent_dim_min = np.argmax(cumsum >= variance_threshold) + 1

    print(f"Nombre de composantes pour {int(variance_threshold*100)}% de variance : {latent_dim_min}")

def train_cvae(cvae, optimizer, scheduler, num_epochs, dataloader_train, dataloader_test, device):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        cvae.train()

        for x_batch, y_batch, labels_batch in dataloader_train:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            labels_batch = labels_batch.to(device)
            A_hat, B_pred, mu, logvar = cvae(x_batch, y_batch, labels_batch)
            loss_batch = cvae.loss(A_hat, x_batch, B_pred, y_batch, mu, logvar, labels_batch)

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            epoch_loss += loss_batch.item()

        avg_loss = epoch_loss / len(dataloader_train)
        train_losses.append(avg_loss)

        # Validation
        cvae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val, labels_val in dataloader_test:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                labels_val = labels_val.to(device)
                A_hat, B_pred, mu, logvar = cvae(x_val, y_val, labels_val)
                loss = cvae.loss(A_hat, x_val, B_pred, y_val, mu, logvar, labels_val)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(dataloader_test)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if (epoch + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate (epoch {epoch+1}): {current_lr:.6e}")

    # Affichage des courbes de loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Courbe de la loss d'entraînement et validation")
    plt.legend()
    plt.show()
 
def save_model(model, optimizer, model_path, input_dim, latent_dim, hidden_dim_list, dropout, scaler_x):
    """
    Sauvegarde le modèle VAE et ses paramètres dans un fichier.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dim_list': hidden_dim_list,
        'dropout': dropout,
        'scaler_x': scaler_x
    }
    torch.save(checkpoint, model_path)
    print(f"Modèle sauvegardé sous {model_path}")

def load_model_cvae(device):
    """
    Charge un modèle VAE sauvegardé et ses paramètres/scalers.
    Retourne : vae, optimizer_state_dict, scaler_x, device
    """
    checkpoint = torch.load("saved_models/cvae_trained_2.pth", map_location="cpu")

    cvae = CVAEIntegrated(
        design_dim=checkpoint["input_dim"],
        spectrum_dim=checkpoint["output_dim"],
        cond_dim=checkpoint["cond_dim"],
        latent_dim=checkpoint["latent_dim"],
        hidden_dims_encoder=checkpoint["hidden_dims_encoder"],
        hidden_dims_decoder=checkpoint["hidden_dims_decoder"],
        hidden_dims_fnn=None,
        dropoutFNN=None,
        hidden_dim_classifier=None,  
        gamma=None, 
        dropoutCVAE=checkpoint["dropout"],
        beta=checkpoint["beta"]
    ).to(device)
    cvae.load_state_dict(checkpoint["model_state_dict"])
    cvae.eval()

    optimizer = torch.optim.Adam(cvae.parameters(), lr=checkpoint["lr"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scaler_x = checkpoint["scaler_x"]
    scaler_z = checkpoint["scaler_z"]

    print("Latent dimension :", checkpoint["latent_dim"])
    print("Hidden dimensions encoder :", checkpoint["hidden_dims_encoder"])
    print("Hidden dimensions decoder :", checkpoint["hidden_dims_decoder"])
    print("Dropout :", checkpoint["dropout"])
    print("Beta :", checkpoint["beta"])
    print("Learning rate :", checkpoint["lr"])

    print("Modèle CVAE chargé avec succès.")
    return cvae, scaler_x, scaler_z

def display_spectrums(n_examples, original, recon):
    """
    Affiche n_examples spectres originaux et reconstruits côte à côte.
    original : array [N, ...] (ex: orig_y)
    recon    : array [N, ...] (ex: recon_y)
    """
    n_cols = 5
    n_rows = n_examples // n_cols
    plt.figure(figsize=(2.2 * n_cols, 2.5 * n_rows))
    for i in range(n_examples):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(original[i], label='Original Y')
        plt.plot(recon[i], label='Recon Y')
        plt.title(f"Ex {i+1}", fontsize=8)
        plt.xticks([])
        plt.yticks([])
        if i % n_cols == 0:
            plt.legend(fontsize=7)
    plt.tight_layout()
    plt.show()

def cvae_reconstruction(cvae, dataloader, device):
    """
    Prend un batch du dataloader, effectue la reconstruction avec le CVAEIntegrated,
    affiche les shapes et premiers exemples de A, B, C originaux/reconstruits,
    et trace les courbes des spectres B (original) et B_pred (prédit).
    Retourne : orig_A, orig_B, orig_C, recon_A, pred_B
    """
    cvae.eval()
    with torch.no_grad():
        A, B, C = next(iter(dataloader))
        A = A.to(device)
        B = B.to(device)
        C = C.to(device)
        A_hat, B_pred, _, _, _ = cvae(A, B, C)
        A_hat_np = A_hat.cpu().numpy()
        B_pred_np = B_pred.cpu().numpy()
        A_np = A.cpu().numpy()
        B_np = B.cpu().numpy()
        C_np = C.cpu().numpy()

    print("A_hat shape:", A_hat_np.shape)
    print("B_pred shape:", B_pred_np.shape)
    print("A_hat[0]:", A_hat_np[0])
    print("B_pred[0]:", B_pred_np[0])
    print("A shape:", A_np.shape)
    print("B shape:", B_np.shape)
    print("A[0]:", A_np[0])
    print("B[0]:", B_np[0])
    print("C shape:", C_np.shape)
    print("C[0]:", C_np[0])

    # Affichage des spectres (B et B_pred) pour les 3 premiers exemples
    n_plot = min(3, B_np.shape[0])
    plt.figure(figsize=(12, 4 * n_plot))
    for i in range(n_plot):
        plt.subplot(n_plot, 1, i+1)
        plt.plot(B_np[i], label='B original')
        plt.plot(B_pred_np[i], label="B prédit", linestyle='--')
        plt.title(f"Spectre exemple {i}")
        plt.xlabel("Longueur d'onde (ou index)")
        plt.ylabel("Intensité")
        plt.legend()
    plt.tight_layout()
    plt.show()

    return A_np, B_np, C_np, A_hat_np, B_pred_np

def cvae_latent_3d(
    cvae, dataloader, index1=3, index2=9, index3=8, color_mode="peak_pos", index4=None, device=None
):
    """
    Affiche une projection 3D de l'espace latent du CVAE pour un batch du dataloader.
    index1, index2, index3 : indices des dimensions latentes à afficher (axes).
    color_mode : "peak_pos" (par défaut), "sum", ou "label" (pour la couleur).
    index4 : indice de la 5e dimension à utiliser pour la taille des points (optionnel).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prendre un batch
    x_batch, y_batch, label_batch = next(iter(dataloader))
    xy_batch = torch.cat([x_batch, y_batch], dim=1).to(device)
    label_batch = label_batch.to(device)

    with torch.no_grad():
        z_mean, _ = cvae.encode(xy_batch, label_batch)  # [N, latent_dim]

    if color_mode == "peak_pos":
        color = y_batch.argmax(dim=1).cpu().numpy()
        color_label = "Position du pic (indice)"
    elif color_mode == "sum":
        color = y_batch.sum(dim=1).cpu().numpy()
        color_label = "Somme(y)"
    elif color_mode == "label":
        color = label_batch.cpu().numpy().squeeze()
        color_label = "Label"
    else:
        color = None
        color_label = ""

    # Taille des points selon la 5e dimension si index4 est fourni
    if index4 is not None:
        size = z_mean[:, index4].detach().cpu().numpy()
        size = np.abs(size)
        ptp = np.ptp(size)
        if ptp < 1e-6:
            size = np.full_like(size, 20.0)  # taille fixe si pas de variance
        else:
            sqrt_size = np.sqrt(size)
            size = 10 + 30 * (sqrt_size - sqrt_size.min()) / (np.ptp(sqrt_size) + 1e-8)
    else:
        size = None

    fig = px.scatter_3d(
        x=z_mean[:, index1].detach().cpu().numpy(),
        y=z_mean[:, index2].detach().cpu().numpy(),
        z=z_mean[:, index3].detach().cpu().numpy(),
        color=color,
        size=size,
        labels={'color': color_label, 'size': f'z[{index4}]' if index4 is not None else None}
    )
    fig.update_layout(title=f"Latent space: z[{index1}], z[{index2}], z[{index3}]"
                                 + (f", size=z[{index4}]" if index4 is not None else ""))
    fig.show()


def cvae_latent_2d(cvae, dataloader, index_x=3, index_y=9, index_histo=8, device=None, label_col=0):
    """
    Affiche des scatterplots 2D du latent z échantillonné, colorés selon la somme des cibles,
    la position du pic, et la valeur du label (colonne label_col de C_batch).
    Compatible CVAEIntegrated : attend A, B, C dans le batch.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_iter = iter(dataloader)
    A_batch, B_batch, C_batch = next(data_iter)
    print("A_batch shape:", A_batch.shape)
    print("B_batch shape:", B_batch.shape)
    print("C_batch shape:", C_batch.shape)
    
    A_batch = A_batch.to(device)
    B_batch = B_batch.to(device)
    C_batch = C_batch.to(device)

    mu, log_var = cvae.encode(A_batch, B_batch, C_batch)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mu + std * eps  # <-- échantillon réel

    # 1. Couleur selon la somme des valeurs cibles
    color_sum = B_batch.sum(dim=1).cpu().numpy()
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        z[:, index_x].detach().cpu().numpy(),
        z[:, index_y].detach().cpu().numpy(),
        c=color_sum, alpha=0.6, cmap='viridis'
    )
    plt.colorbar(sc, label="Somme(y)")
    plt.title("z ~ N(0, I) ? (Somme des valeurs cibles)")
    plt.xlabel(f"z[{index_x}]")
    plt.ylabel(f"z[{index_y}]")
    plt.grid(True)
    plt.show()

    # 2. Couleur selon la position du pic (indice du max)
    color_peak_pos = B_batch.argmax(dim=1).cpu().numpy()
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        z[:, index_x].detach().cpu().numpy(),
        z[:, index_y].detach().cpu().numpy(),
        c=color_peak_pos, alpha=0.6, cmap='viridis'
    )
    plt.colorbar(sc, label="Position du pic (indice)")
    plt.title("z ~ N(0, I) ? (Position du pic)")
    plt.xlabel(f"z[{index_x}]")
    plt.ylabel(f"z[{index_y}]")
    plt.grid(True)
    plt.show()

    # 3. Couleur selon la valeur du label (une seule colonne)
    color_label = C_batch[:, label_col].cpu().numpy()
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        z[:, index_x].detach().cpu().numpy(),
        z[:, index_y].detach().cpu().numpy(),
        c=color_label, alpha=0.6, cmap='tab10' if len(np.unique(color_label)) < 10 else 'viridis'
    )
    plt.colorbar(sc, label=f"Label (C[:, {label_col}])")
    plt.title("z ~ N(0, I) ? (Label)")
    plt.xlabel(f"z[{index_x}]")
    plt.ylabel(f"z[{index_y}]")
    plt.grid(True)
    plt.show()

    print("Variance moyenne de z :", z.var(dim=0).mean().item())
    print("Moyenne de z :", z.mean(dim=0).detach().cpu().numpy())
    plt.hist(z[:, index_histo].detach().cpu().numpy(), bins=50)
    plt.title(f"Histogramme de z[{index_histo}]")
    plt.show()

def vae_latent_interpolation(vae, latent_dim=32, dim_x=4, axis=0, z_min=-2, z_max=2, steps=15, fixed_z=None, device=None):
    """
    Affiche l'effet de l'interpolation sur un axe latent donné.
    - vae : modèle VAE
    - dim_x : dimensions de X dans la sortie du décodeur
    - axis : indice de l'axe latent à faire varier
    - z_min, z_max : bornes de variation de l'axe
    - steps : nombre de points à générer
    - fixed_z : vecteur latent de base (sinon zeros)
    - device : cpu ou cuda
    """
    vae.eval()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if fixed_z is None:
        z_base = torch.zeros(latent_dim, device=device)
    else:
        z_base = torch.tensor(fixed_z, dtype=torch.float32, device=device)

    z_values = np.linspace(z_min, z_max, steps)
    decoded_y = []

    with torch.no_grad():
        for val in z_values:
            z = z_base.clone()
            z[axis] = val
            out = vae.decode(z.unsqueeze(0)).cpu().numpy()
            y = out[:, dim_x:]  # On suppose que la sortie est [X|Y]
            decoded_y.append(y[0])

    plt.figure(figsize=(10, 2.5 * steps // 5))
    for i, y in enumerate(decoded_y):
        plt.subplot(steps // 5, 5, i + 1)
        plt.plot(y)
        plt.title(f"z[{axis}]={z_values[i]:.2f}", fontsize=8)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.suptitle(f"Interpolation sur z[{axis}] de {z_min} à {z_max}", y=1.02)
    plt.show()

def print_top_latent_std(std_tensor, top_n=3):
    """
    Affiche les valeurs de std du latent dans l'ordre décroissant
    et retourne les indices des top_n plus grandes valeurs.
    """
    std_np = std_tensor.detach().cpu().numpy()
    sorted_indices = std_np.argsort()[::-1]
    print("Std of z_train (sorted):")
    for idx in sorted_indices:
        print(f"z[{idx}] : {std_np[idx]:.4f}")
    top_indices = sorted_indices[:top_n]
    print(f"\nIndices des {top_n} plus grandes std :", top_indices)
    return top_indices

def cvae_latent_umap(
    cvae, dataloader,scaler_z, color_mode="peak_pos", device=None, n_neighbors=15, min_dist=0.1, metric="euclidean"
):
    """
    Affiche une projection 3D UMAP de l'espace latent du CVAEIntegrated pour un batch du dataloader.
    color_mode : "peak_pos", "sum", "label_f" (fréquence), ou "label_n" (n_eff) pour la couleur.
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prendre un batch
    A_batch, B_batch, C_batch = next(iter(dataloader))
    A_batch = A_batch.to(device)
    B_batch = B_batch.to(device)
    C_batch = C_batch.to(device)
    
    with torch.no_grad():
        z_mean, _ = cvae.encode(A_batch, B_batch, C_batch)  # [N, latent_dim]
        z_np = z_mean.cpu().numpy()

    C_batch = scaler_z.inverse_transform(C_batch.cpu().numpy()) if hasattr(scaler_z, "transform") else C_batch

    if color_mode == "peak_pos":
        color = B_batch.argmax(dim=1).cpu().numpy()
        color_label = "Position du pic (indice)"
    elif color_mode == "sum":
        color = B_batch.sum(dim=1).cpu().numpy()
        color_label = "Somme(y)"
    elif color_mode == "label_f":
        color = C_batch[:, 0]
        color_label = "Fréquence du pic"
    elif color_mode == "label_n":
        color = C_batch[:, 1]
        color_label = "n_eff"
    else:
        color = None
        color_label = ""

    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    z_umap = reducer.fit_transform(z_np)

    fig = px.scatter_3d(
        x=z_umap[:, 0],
        y=z_umap[:, 1],
        z=z_umap[:, 2],
        color=color,
        labels={'color': color_label}
    )
    fig.update_layout(title="Projection UMAP 3D de l'espace latent")
    fig.show()


def cvae_latent_umap_static(
    cvae, dataloader, scaler_z, color_mode="peak_pos", device=None, 
    n_neighbors=15, min_dist=0.1, metric="euclidean", streamlit_show=False
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prendre un batch
    A_batch, B_batch, C_batch = next(iter(dataloader))
    A_batch = A_batch.to(device)
    B_batch = B_batch.to(device)
    C_batch = C_batch.to(device)

    with torch.no_grad():
        z_mean, _ = cvae.encode(A_batch, B_batch, C_batch)
        z_np = z_mean.cpu().numpy()

    C_batch = scaler_z.inverse_transform(C_batch.cpu().numpy())

    # Fréquences pour la légende si besoin
    frequencies = np.linspace(171309976000000, 222068487407407, B_batch.shape[1])

    # Couleurs
    if color_mode == "peak_pos":
        color = B_batch.argmax(dim=1).cpu().numpy()
        color_label = "Position du pic (Hz)"
        # Remplace les indices par les fréquences correspondantes pour la légende
        color_freq = frequencies[color]
    elif color_mode == "sum":
        color = B_batch.sum(dim=1).cpu().numpy()
        color_label = "Somme(y)"
        color_freq = None
    elif color_mode == "label_f":
        color = C_batch[:, 0]
        color_label = "Fréquence du pic"
        color_freq = None
    elif color_mode == "label_n":
        color = C_batch[:, 1]
        color_label = "n_eff"
        color_freq = None
    else:
        color = None
        color_label = ""
        color_freq = None

    # UMAP
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    z_umap = reducer.fit_transform(z_np)

    # Affichage matplotlib
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    if color_mode == "peak_pos":
        p = ax.scatter(z_umap[:, 0], z_umap[:, 1], z_umap[:, 2], c=color_freq, cmap='plasma', s=20)
        cbar = fig.colorbar(p, ax=ax, label=color_label)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
    else:
        p = ax.scatter(z_umap[:, 0], z_umap[:, 1], z_umap[:, 2], c=color, cmap='viridis', s=20)
        fig.colorbar(p, ax=ax, label=color_label)

    ax.set_title("Projection UMAP 3D de l'espace latent")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")

    if streamlit_show:
        st.pyplot(fig)
    else:
        plt.show()

def hybrid_model_evaluation(cvae, dataloader_train, device, f_desired, n_desired, tol_f=0.05, tol_n=0.05, n_samples=20000, scaler_x=None, scaler_z=None, saving=False):

    # Extraction des vecteurs latents (mu) pour modéliser la distribution latente (préparation à la génération synthétique)
    # ============================================================
    z_list = []
    cvae.eval()
    with torch.no_grad():
        for A,B,C in dataloader_train:
            A, B, C = A.to(device), B.to(device), C.to(device)
            mu, logvar = cvae.encode(A,B,C)
            z_list.append(mu.cpu())  # stocke les mu (sans reparam.)
    # Concatène tous les z en un seul tensor
    z_train = torch.cat(z_list, dim=0)  # shape [N, latent_dim]
    mean = z_train.mean(dim=0)  # centre
    std = z_train.std(dim=0)    # écart-type par dimension
    # Matrice de covariance
    cov = torch.cov(z_train.T)  # shape (latent_dim, latent_dim)
    mean = mean.to(device)
    std = std.to(device)
    cov = cov.to(device)

    # Génération et filtrage par densité de probabilité
    # ============================================================
    label_value = [f_desired, n_desired]  # le label que tu veux générer
    label_value_scaled = scaler_z.transform(np.array([label_value]))
    label_value_scaled = torch.tensor(label_value_scaled, dtype=torch.float32).to(device)

    mvn = MultivariateNormal(mean, covariance_matrix=cov)
    z = mvn.sample((n_samples,)).to(device)

    log_probs = mvn.log_prob(z)
    threshold = torch.quantile(log_probs, 0.01)  # Garde les 99% les plus probables
    mask = log_probs > threshold
    z_filtered = z[mask]

    label_tensor = label_value_scaled.expand(z_filtered.shape[0], -1)

    # Générer les données avec le décodeur à partir des z filtrés
    with torch.no_grad():
        A_hat_gen, B_pred_gen = cvae.decode(z_filtered, label_tensor)  # [n_samples_filtrés, input_dim]

    A_hat_gen_denorm = scaler_x.inverse_transform(A_hat_gen.cpu().numpy())

    # Analyse des pics dans les spectres générés
    # ============================================================
    window = gaussian(M=21, std=5)
    window /= window.sum()
    frequencies = np.linspace(171309976000000, 222068487407407, 5000)
    c = 299792458  # vitesse de la lumière

    x_1pic, y_1pic, f_1pic, n_eff_1pic = [], [], [], []

    for i in range(len(B_pred_gen)):
        y = B_pred_gen[i].cpu().numpy()  # Convertit le spectre en numpy array
        y_smooth = filtfilt(window, [1], y)
        peaks, _ = find_peaks(y_smooth, height=0.1, prominence=0.05, distance=10)
        n_peaks = len(peaks)
        x = A_hat_gen_denorm[i]

        if n_peaks == 1:
            idx_max = np.argmax(y)
            f = frequencies[idx_max]
            n_eff = (c * x[3]) / (2 * np.pi * f)

            x_1pic.append(x)
            y_1pic.append(y)
            f_1pic.append(f)
            n_eff_1pic.append(n_eff)

    # Convertir en arrays numpy si ce n'est pas déjà fait
    x_1pic = np.array(x_1pic)
    y_1pic = np.array(y_1pic)
    f_1pic = np.array(f_1pic)
    n_eff_1pic = np.array(n_eff_1pic)

    # Masque de tolérance
    mask_1pic = (np.abs(n_eff_1pic - n_desired) <= tol_n)

    x_1pic_filt = x_1pic[mask_1pic]
    y_1pic_filt = y_1pic[mask_1pic]
    f_1pic_filt = f_1pic[mask_1pic]
    n_eff_1pic_filt = n_eff_1pic[mask_1pic]


    # Filtrage par tolérance sur la fréquence
    # ============================================================

    # Masque de tolérance
    mask_1pic_2 = (np.abs(f_1pic_filt - f_desired) <= tol_f)

    x_1pic_filt_2 = x_1pic_filt[mask_1pic_2]
    y_1pic_filt_2 = y_1pic_filt[mask_1pic_2]
    f_1pic_filt_2 = f_1pic_filt[mask_1pic_2]
    n_eff_1pic_filt_2 = n_eff_1pic_filt[mask_1pic_2]

    # Filtrage par contrainte de fabrication
    # ============================================================
    result1 = (1 - x_1pic_filt_2[:, 1]) * x_1pic_filt_2[:, 2]
    result2 = x_1pic_filt_2[:, 1] * x_1pic_filt_2[:, 2]
    mask_fab = (result1 >= 50) & (result2 >= 50)

    # Application du mask
    x_1pic_filt_3 = x_1pic_filt_2[mask_fab]
    y_1pic_filt_3 = y_1pic_filt_2[mask_fab]
    f_1pic_filt_3 = f_1pic_filt_2[mask_fab]
    n_eff_1pic_filt_3 = n_eff_1pic_filt_2[mask_fab]


    if saving == True:

        df_params = pd.DataFrame(x_1pic_filt_3, columns=["w", "DC", "pitch", "k"])
        df_params.to_csv("results/SWG_waveguide_geometry.csv", index=False)

        # 2. model_spectres.csv (5000 lignes, N colonnes)
        # Chaque colonne correspond à un spectre, chaque ligne à un point du spectre (fréquence)
        y_1pic_filt_3_T = y_1pic_filt_3.T  # shape (5000, N)
        spec_cols = [f"spectre_{i}" for i in range(y_1pic_filt_3_T.shape[1])]
        df_spectres = pd.DataFrame(y_1pic_filt_3_T, columns=spec_cols)
        df_spectres.to_csv("results/SWG_waveguide_spectra.csv", index=False)

        # 3. model_labels.csv (N, 2) avec header (f_max, n_eff)
        df_labels = pd.DataFrame({
            "f_max": f_1pic_filt_3,
            "n_eff": n_eff_1pic_filt_3
        })
        df_labels.to_csv("results/SWG_waveguide_labels.csv", index=False)

        print("Fichiers CSV exportés dans results/ : SWG_waveguide_geometry.csv, SWG_waveguide_spectra.csv, SWG_waveguide_labels.csv")

    return x_1pic_filt_3, y_1pic_filt_3, f_1pic_filt_3, n_eff_1pic_filt_3




