import torch
from scipy.optimize import minimize


def get_optimal_radius_tau(model, val_loader, criterion, device):

    def scaled_cross_entropy_loss(embeds, labels, tau):
        embeds, labels = embeds.to(device), labels.to(device)
        tau = torch.tensor(tau, dtype=torch.float32, device=device)
        scaled_embeds = embeds / tau
        logits = model.module.decoder(scaled_embeds)
        loss = criterion(logits, labels)
        return loss.item()

    embeddings_list, labels_list = [], []

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            embeds = model.module.embed(x)
        embeddings_list.append(embeds.cpu())
        labels_list.append(y.cpu())

    embeds = torch.cat(embeddings_list)
    labels = torch.cat(labels_list)

    res = minimize(
        lambda tau: scaled_cross_entropy_loss(embeds, labels, tau),
        1.0,
        bounds=[(0.05, 5.0)],
        options={"eps": 0.01},
    )

    optimal_tau = res.x[0]

    return optimal_tau


def get_optimal_confidence_tau(model, val_loader, criterion, device):

    def scaled_cross_entropy_loss(logits, labels, tau):
        scaled_logits = logits / tau
        loss = criterion(scaled_logits, labels)
        return loss.item()

    logits_list, labels_list = [], []

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            embeds = model.module.embed(x)
            logits = model.module.decoder(embeds)
        logits_list.append(logits.cpu())
        labels_list.append(y.cpu())

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    res = minimize(
        lambda tau: scaled_cross_entropy_loss(logits, labels, tau),
        1.0,
        method="L-BFGS-B",
        bounds=[(0.05, 3.0)],
    )

    optimal_tau = res.x[0]

    return optimal_tau
