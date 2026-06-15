import torch

def normalize_self_attention(R):
    """
    Applies row-normalization to a self-relevance matrix (Eq. 8-9 from Chefer et al.).
    Removes identity diagonal influence, normalizes rows, and restores 1s on the diagonal.
    """
    R_clone = R.clone()
    diag_indices = torch.arange(R_clone.shape[0], device=R.device)
    
    # 1. Remove identity diagonal influence
    R_clone[diag_indices, diag_indices] = 0
    
    # 2. Normalize by row sum
    row_sums = R_clone.sum(dim=-1, keepdim=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    R_norm = R_clone / row_sums
    
    # 3. Restore 1s on the diagonal
    R_norm[diag_indices, diag_indices] = 1.0
    return R_norm


def attention_rollout_chefer(
    cross_A_maps, cross_A_grads,  # Block A: Text -> Image (M, N)
    self_B_maps,  self_B_grads,   # Block B: Text -> Text  (M, M)
    cross_C_maps, cross_C_grads,  # Block C: Image -> Text (N, M)
    R_ii_encoder=None, 
    R_tt_encoder=None,   
    device='cpu'
):
    """
    Computes the essential Chefer relevance matrix required for 3D visualization.
    Traverses decoder layers backwards, blending raw attention with gradients.
    
    Dimensions:
    - M: Total Text Queries / Tokens (num_queries_total)
    - N: Total 3D Points in Point Cloud (num_points)
    """
    # Deduce dimensions from the first collected layer (already sliced at batch index 0)
    num_queries_total = cross_A_maps[0].shape[1]  # M = Q + T
    num_points = cross_A_maps[0].shape[2]         # N
    
    # Initialize the 3 core relevance accumulators
    R_tt = R_tt_encoder.to(device).clone() if R_tt_encoder is not None else torch.eye(num_queries_total, device=device)
    R_ii = R_ii_encoder.to(device).clone() if R_ii_encoder is not None else torch.eye(num_points, device=device)
    R_ti = torch.zeros((num_queries_total, num_points), device=device)

    # Step backward through the decoder layers (from last layer up to the first)
    for l in reversed(range(len(cross_A_maps))):
        
        # =====================================================================
        # STEP 1: BLOCCO C - Cross-Attention Inversa (Image -> Text)
        # =====================================================================
        A_C = cross_C_maps[l].to(device)  # [H, N, M] 
        G_C = cross_C_grads[l].to(device)  # [H, N, M]
        # Gradient-weighted attention: clamp negative gradients to 0, average over heads
        A_bar_C = torch.clamp(A_C * G_C, min=0).mean(dim=0)  # Shape: (N, M)
        # print(f"Layer {l} - Max Grad: {G_C.max().item()}, Min Grad: {G_C.min().item()}")
        # print(f"Layer {l} - Elementi maggiori di zero in A_bar_C: {(A_bar_C > 0).sum().item()} su {A_bar_C.numel()}")
        # Update Self-Image (Eq. 11): Image absorbs current Text relevance state
        R_ii = R_ii + torch.matmul(A_bar_C, R_ti)
        
        # =====================================================================
        # STEP 2: BLOCCO B - Self-Attention del Testo/Query (Text -> Text)
        # =====================================================================
        A_B = self_B_maps[l].to(device)   # [H, M, M]
        G_B = self_B_grads[l].to(device)  # [H, M, M]
        A_bar_B = torch.clamp(A_B * G_B, min=0).mean(dim=0)  # Shape: (M, M)
        # print(f"Layer {l} - Max Grad: {G_B.max().item()}, Min Grad: {G_B.min().item()}")
        # print(f"Layer {l} - Elementi maggiori di zero in A_bar_B: {(A_bar_B > 0).sum().item()} su {A_bar_B.numel()}")
        # Update Self-Text (Eq. 6): Text queries mix their internal history
        R_tt = R_tt + torch.matmul(A_bar_B, R_tt)
        
        # Propagate Cross Relevance (Eq. 7): Visual data shifts alongside text tokens
        R_ti = R_ti + torch.matmul(A_bar_B, R_ti)

        # =====================================================================
        # STEP 3: BLOCCO A - Cross-Attention Principale (Text -> Image)
        # =====================================================================
        A_A = cross_A_maps[l].to(device)   
        G_A = cross_A_grads[l].to(device)  
        A_bar_A = torch.clamp(A_A * G_A, min=0).mean(dim=0)  # Shape: (M, N)
        # print(f"Layer {l} - Max Grad: {G_A.max().item()}, Min Grad: {G_A.min().item()}")
        # print(f"Layer {l} - Elementi maggiori di zero in A_bar_A: {(A_bar_A > 0).sum().item()} su {A_bar_A.numel()}")
        # Normalize self histories to compute the core cross contribution
        R_tt_norm = normalize_self_attention(R_tt)
        R_ii_norm = normalize_self_attention(R_ii)
        
        # Generate new Cross-Relevance (Eq. 10): Core alignment layer
        cross_A_contrib = torch.matmul(R_tt_norm.t(), A_bar_A)
        cross_A_contrib = torch.matmul(cross_A_contrib, R_ii_norm)
        
        R_ti = R_ti + cross_A_contrib

    # Return the final alignment matrix (M x N)
    return R_ti