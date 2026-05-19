import torch


def compute_head_weights(gradients, reg_type="L1"):
    # gradients shape: (Heads, Q, P)
    if reg_type == "L1":
        # Somma dei valori assoluti per ogni testa
        head_importance = gradients.abs().sum(dim=(1, 2)) 
    elif reg_type == "L2":
        # Radice della somma dei quadrati
        head_importance = torch.sqrt((gradients**2).sum(dim=(1, 2)))
    
    # Normalizzazione (w = GRP / sum(GRP))
    w = head_importance / (head_importance.sum() + 1e-8)
    return w

def attention_rollout(cross_attention_maps, cross_attention_grads, device='cpu',alpha=0.5):
    ''''
    Peforms gradient-based attention rollout (reference: https://arxiv.org/pdf/2504.19414).
    Inputs:
    - cross_attention_maps: dict of attention maps from each layer (B, Heads, Q+T, N) 
    - cross_attention_grads: dict of gradients for each attention map (B, Heads, Q+T, N)
    - device: device to perform computations on
    - alpha: residual ratio for rollout update
    where
    - B: batch size
    - Heads: number of attention heads
    - Q: number of queries (pred mask tokens)
    - T: number of text tokens
    - N: number of points in the point cloud
    Output:
    - A_rollout: final attention map after rollout (Q+T, N)
    '''

    for i in range(len(cross_attention_maps)):
        A_l = cross_attention_maps[i][0].to('cpu') # Prendi il batch 0 -> (Heads, Q, P)
        G_l = cross_attention_grads[i][0]   # (Heads, Q, P)
        
        # 1. Calcola i pesi delle teste per questo layer
        w = compute_head_weights(G_l, reg_type="L1") # -> (Heads,)
        
        # 2. Pesa le teste dell'attenzione (A_weighted = A_l * W)
        # Espandiamo w per la moltiplicazione: (Heads, 1, 1)
        w_reshaped = w.view(-1, 1, 1)
        a_weighted = (A_l * w_reshaped).sum(dim=0) # Media pesata delle teste -> (Q+T, N)
        
        # Rollout Update (A_rollout = A_rollout * A_weighted + alpha * I)
        # a_rollout = torch.matmul(a_rollout, a_weighted) + alpha * torch.eye(num_queries).to(self.device)
        if i == 0:
            A_rollout = a_weighted
        else:
            # Questo simula il flusso di informazioni attraverso i layer
            A_rollout = a_weighted + alpha * A_rollout #[Q+T, N]
    return A_rollout.to(device)