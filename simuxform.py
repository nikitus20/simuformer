import argparse
import torch
import gc
import time
import numpy as np
import pickle
import datetime
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize  # Correct import
import tqdm
import math
import random

class SimuXForm:
    def __init__(self, args, beta, V=None, BF=None):
        batch, ntokens, dmodel, n_heads = args.batch, args.ntokens, args.dmodel, args.n_heads

        self.args = args
        self.beta = beta
        self.device = args.device

        # Check for use_softmax, add default if missing from args for safety
        use_softmax = getattr(args, 'use_softmax', True) 

        # Normalize step to prevent moves larger than 0.1 radians
        if not use_softmax and not args.rawstep:
            # Ensure beta is a scalar float for math functions if it's a tensor
            beta_val = beta.item() if isinstance(beta, torch.Tensor) else beta
            denom = max((np.sqrt(beta_val) * np.exp(beta_val - 0.5), 1))
            eta = args.step / denom if denom > 0 else args.step # Avoid division by zero
            print(f'Corrected step = {eta:.4f}')
        else:
            eta = args.step

        self.eta = eta
        self.n_heads = n_heads

        # Initialize tensors on the correct device
        with torch.no_grad():
            self.r = torch.ones(batch, ntokens, device=self.device)
            self.IP = torch.zeros(batch, ntokens, ntokens, device=self.device)
            self.SM = torch.zeros(batch, ntokens, ntokens, device=self.device)
            self.M2 = torch.randn(batch, ntokens, dmodel, device=self.device)

        # Random Bilinear form (Q^T K thing) & V matrix
        self.BF = [None] * n_heads 
        self.V = [None] * n_heads

        # Generate matrices per head (these will be on self.device)
        for i in range(n_heads):
            head_V = V[i] if V is not None and len(V) == n_heads else None
            head_BF = BF[i] if BF is not None and len(BF) == n_heads else None
            # Ensure precomputed are on the correct device if provided
            if head_V is not None: head_V = head_V.to(self.device)
            if head_BF is not None: head_BF = head_BF.to(self.device)
            self.generate_KQV(i, precomputed_V=head_V, precomputed_BF=head_BF)

        # Initialize particles on the sphere on the correct device
        with torch.no_grad():
            self.M = torch.nn.functional.normalize(self.M2, dim=2)
            self.M2 = torch.empty_like(self.M) # Re-create M2 on the correct device

        # Now analyze eigenvectors if requested
        if self.args.analyze_evecs:
             self.analyze_eigenvectors()


    def generate_KQV(self, i, precomputed_V=None, precomputed_BF=None):
        args = self.args
        batch, ntokens, dmodel = args.batch, args.ntokens, args.dmodel
        device = self.device

        # Reuse precomputed if noanneal=2 and they are provided (already on device)
        if args.noanneal == 2 and precomputed_BF is not None and precomputed_V is not None:
            print(f'[DEBUG Head {i}] noanneal=2, reusing provided BF and V.')
            self.BF[i] = precomputed_BF
            self.V[i] = precomputed_V
            return
        
        # Reuse already generated if noanneal=2 and they exist (already on device)
        if args.noanneal == 2 and self.BF[i] is not None and self.V[i] is not None:
             print(f'[DEBUG Head {i}] noanneal=2, reusing existing BF and V.')
             return

        pp = 1 if args.noanneal > 0 else batch

        with torch.no_grad():
            # Generate BF (K^T Q) based on randomKQ flag (on self.device)
            if args.randomKQ == 1: 
                BF1 = torch.randn(pp, dmodel, dmodel, device=device)
                BF2 = torch.randn(pp, dmodel, dmodel, device=device)
                bf_i = BF1 @ BF2 / math.sqrt(dmodel)
                normV = math.sqrt(dmodel)
            elif args.randomKQ == 2: 
                BF1 = torch.randn(pp, dmodel, dmodel, device=device)
                bf_i = (BF1 + BF1.transpose(1, 2)) / math.sqrt(2)
                normV = math.sqrt(dmodel)
            elif args.randomKQ == 3: # Spec: orthogonal-like. Impl: same as 2.
                BF1 = torch.randn(pp, dmodel, dmodel, device=device)
                bf_i = (BF1 + BF1.transpose(1, 2)) / math.sqrt(2)
                normV = math.sqrt(dmodel + 1) # Only diff from mode 2 is this normV
            elif args.randomKQ == 4: # GPT-2 style init (std=0.02)
                bf_i = torch.randn(pp, dmodel, dmodel, device=device) * 0.02
                normV = 1.0 # Define normV=1.0 so randomV=2/3 use BF/-BF directly
            else: # Identity (randomKQ=0)
                bf_i = torch.eye(dmodel, device=device).unsqueeze(0).expand(pp, -1, -1)
                normV = 1.0

            self.BF[i] = bf_i

            # Generate V based on randomV flag (on self.device)
            if args.randomV == 1: 
                v_i = torch.randn(pp, dmodel, dmodel, device=device) / math.sqrt(dmodel)
            elif args.randomV == 2: 
                # Need BF[i] to be generated first
                if self.BF[i] is None: # Should not happen if called in order
                    raise ValueError("BF[i] must be generated before V[i] can be derived in mode 2.")
                v_i = self.BF[i] / normV 
            elif args.randomV == 3: 
                if self.BF[i] is None:
                    raise ValueError("BF[i] must be generated before V[i] can be derived in mode 3.")
                v_i = -self.BF[i] / normV
            elif args.randomV == 4: # GPT-2 style init (std=0.02)
                v_i = torch.randn(pp, dmodel, dmodel, device=device) * 0.02
            else: # Identity (randomV=0)
                v_i = torch.eye(dmodel, device=device).unsqueeze(0).expand(pp, -1, -1)
            
            self.V[i] = v_i


    def analyze_eigenvectors(self):
        # Optional: Analyze eigenvectors and compute cosine similarity
        # Removed file writing. Just prints to console now.
        if not self.args.analyze_evecs:
            return

        main_evecs = []
        print("[INFO] Analyzing V matrix eigenvectors...")
        for i in range(self.n_heads):
            if self.V[i] is None:
                print(f"[WARN] V matrix for head {i} is None. Skipping eigenvector analysis for this head.")
                continue
            # Ensure V is on CPU for numpy/linalg operations if they fail on GPU, or check device compatibility
            # Assuming V is on self.device. torch.linalg.eigh should work.
            try:
                 with torch.no_grad():
                    # Handle batch dimension if present (pp=batch). Analyze first batch element? Or mean?
                    # Current V might have shape [pp, d, d]. Taking first element [0].
                    V_matrix_head_i = self.V[i][0] if self.V[i].dim() == 3 else self.V[i]
                    evals, evecs = torch.linalg.eigh(V_matrix_head_i) # Use Hermitian eigh for symmetric/hermitian
                    main_evec = evecs[:, -1] # Eigenvector for largest eigenvalue
                    main_evecs.append(main_evec)
            except Exception as e:
                 print(f"[ERROR] Eigenvector analysis failed for head {i}: {e}")
                 return # Stop analysis if one head fails

        if not main_evecs:
            print("[WARN] No eigenvectors could be computed.")
            return

        # Stack and normalize eigenvectors
        try:
            with torch.no_grad():
                stacked_evecs = torch.stack(main_evecs) # Shape [n_heads, dmodel]
                # Ensure normalization is done correctly even with complex numbers if eigh returns them
                normalized_evecs = torch.nn.functional.normalize(stacked_evecs.float(), dim=1) 
                cosine_similarity_matrix = normalized_evecs @ normalized_evecs.T
            
            # Print results instead of saving to file
            print(f"--- Eigenvector Cosine Similarity (V:{self.args.randomV}, KQ:{self.args.randomKQ}) ---")
            # Format output nicely
            print(np.array_str(cosine_similarity_matrix.cpu().numpy(), precision=3, suppress_small=True))
            print("----------------------------------------------------------")

        except Exception as e:
             print(f"[ERROR] Failed to compute/print cosine similarity: {e}")


    def update_IP(self):
        # Compute inner products M M^T
        with torch.no_grad():
            # Ensure M is contiguous for bmm
            self.IP = torch.bmm(self.M.contiguous(), self.M.transpose(1, 2))

    def check_stats(self):
        with torch.no_grad():
            self.update_IP()
            batch, ntokens = self.args.batch, self.args.ntokens
            device = self.device

            # Density calculation
            if self.args.plotdim: # Use eigenvalues for density? Seems unconventional.
                try:
                    # eigvalsh is for Hermitian matrices. IP should be symmetric positive semi-definite.
                    eigv = torch.linalg.eigvalsh(self.IP)
                    # Density as fraction of positive eigenvalues (above threshold)?
                    dens = (eigv > 1e-3).sum(dim=1).float().mean().item() # Mean over batch
                except Exception as inst:
                    print(f'[ERROR] linalg.eigvalsh failed during check_stats: {inst}')
                    dens = 0.0
            else:
                # Density as fraction of pairs with high inner product (>0.999)
                mask = ~torch.eye(ntokens, device=device, dtype=torch.bool).unsqueeze(0) # Mask diagonal
                if self.args.adjacent:
                     # Check only adjacent tokens (need careful masking)
                     # This mask seems wrong: torch.eye(N, N+1)[:, 1:] gives identity shifted right
                     # Let's create a mask for adjacent pairs (i, i+1)
                     adj_mask = torch.zeros(ntokens, ntokens, dtype=torch.bool, device=device)
                     for k in range(ntokens - 1):
                         adj_mask[k, k+1] = True
                     mask = adj_mask.unsqueeze(0) # Add batch dim, shape [1, N, N]

                inner_prods_off_diag = torch.masked_select(self.IP, mask)
                if inner_prods_off_diag.numel() > 0:
                     # Sum over all masked elements (all pairs in batch), divide by total number
                     dens = (inner_prods_off_diag > 0.999).float().sum().item() / inner_prods_off_diag.numel()
                else:
                     dens = 0.0 # Avoid division by zero if ntokens=1 or only diagonal selected


            # Median cluster size calculation (optional)
            median_clust = 0.0
            if self.args.cluster_sizes:
                list_clust = []
                for i in range(batch):
                    try:
                        # Adjacency matrix based on high IP
                        T1 = (self.IP[i] > 0.9999).float() 
                        # Rank of adjacency matrix might relate to connected components / cluster count? Seems complex.
                        # Maybe approximate number of clusters? Or size of largest cluster?
                        # Let's stick to the original code's metric (rank) for now.
                        nr_clust = torch.linalg.matrix_rank(T1).item() 
                        list_clust.append(nr_clust)
                    except Exception as inst:
                         print(f'[ERROR] Cluster calculation failed for batch {i}: {inst}')
                         # Skip this batch element or append a default value? Append 0 for now.
                         list_clust.append(0)

                if list_clust:
                    median_clust = np.median(list_clust)

        return dens, median_clust

    def step(self, current_time): # Renamed from step_1
        args = self.args
        batch, ntokens, dmodel, n_heads = args.batch, args.ntokens, args.dmodel, args.n_heads
        device = self.device
        eta = self.eta # Use pre-calculated eta

        # Check for use_softmax, add default if missing from args for safety
        use_softmax = getattr(args, 'use_softmax', True) 

        with torch.no_grad():
            # --- Sequential Update Structure ---
            # Start with current M for this step. M_current will be updated sequentially by each head.
            M_current = self.M.clone() 

            for i in range(self.n_heads):
                # Get head-specific matrices (BF and V)
                bf_i = self.BF[i] 
                v_i = self.V[i]   
                # Expand shared matrices (pp=1) to batch dimension if needed
                if bf_i is not None and bf_i.shape[0] == 1 and batch > 1: bf_i = bf_i.expand(batch, -1, -1)
                if v_i is not None and v_i.shape[0] == 1 and batch > 1: v_i = v_i.expand(batch, -1, -1)

                # --- Attention Calculation ---
                # 1. Calculate QK Scores (Apply K^T Q transform if specified)
                if args.randomKQ > 0 and bf_i is not None: 
                    # M2 = M_current @ BF[i]
                    M2_scores = torch.einsum('bni,bij->bnj', M_current, bf_i)
                else: # randomKQ == 0 (Identity K^T Q)
                    M2_scores = M_current # No transform needed

                # Calculate raw scores (Attention Logits): M2_scores @ M_current^T
                SM = torch.einsum('bni,bji->bnj', M2_scores, M_current) # [b, n, n]
                SM *= self.beta # Scale by beta

                # 2. Apply Softmax / Normalization
                if use_softmax:
                    if args.sequential: # Apply causal mask (lower triangular)
                        mask = torch.tril(torch.ones(ntokens, ntokens, device=device, dtype=torch.bool)).unsqueeze(0) # [1, n, n]
                        SM = SM.masked_fill(~mask, float('-inf')) # Mask upper triangle FOR NON-DIAGONAL
                    
                    attn_weights = torch.nn.functional.softmax(SM, dim=2) # [b, n, n]
                
                else: # Use simple scaled exp normalization
                    attn_weights = torch.exp(SM)
                    # Normalize rows (or cols?) - original code divided by ntokens
                    # Let's assume row normalization similar to softmax, sum exp first
                    # attn_weights = attn_weights / attn_weights.sum(dim=2, keepdim=True).clamp(min=1e-9) # More stable normalization
                    # Sticking to original code's normalization:
                    attn_weights /= ntokens 

                # --- Value Mixing & Update ---
                # 3. Weighted Sum (Value Mixing): attn_weights @ M_current
                # M2_values = torch.bmm(attn_weights, M_current) # [b, n, n] @ [b, n, d] -> [b, n, d]
                M2_values = torch.einsum('bnj,bjd->bnd', attn_weights, M_current) # Intermediate value vector

                # 4. Pre-LayerNorm Logic (Update scaling factor r if using preln)
                if args.norm == "preln":
                    # Project M2_values onto M_current
                    dot = torch.einsum('bnd,bnd->bn', M2_values, M_current) # Elementwise product sum -> [b, n]
                    # Update scaling factors 'r'
                    self.r.add_( (eta / n_heads) * dot ) # In-place add is efficient
                    # Avoid r becoming zero or negative if updates are large/negative? Clamp maybe?
                    # torch.clamp_(self.r, min=1e-6) # Optional: prevent issues with division

                # 5. Apply Update with Scaling (based on norm type)
                if args.norm == "preln":
                     # Use scaling factor r
                     inv_r = torch.reciprocal(self.r).unsqueeze(-1) # [b, n, 1], use reciprocal for potential speed/stability
                     M_update_delta = M2_values * (eta / n_heads) * inv_r
                elif args.norm == "postln":
                     # No scaling factor 'r', just standard update scaled by eta/n_heads
                     M_update_delta = M2_values * (eta / n_heads)
                else: 
                     # Default or unknown norm type? Fallback to postln behavior.
                     print(f"[WARN] Unknown norm type: {args.norm}. Defaulting to postln behavior in step.")
                     M_update_delta = M2_values * (eta / n_heads)

                # Calculate the state before the V transform: M_current + update_delta
                M_intermediate = M_current + M_update_delta

                # 6. Apply V transform (Output Projection) if specified
                if args.randomV > 0 and v_i is not None:
                    # M_next_head_input = torch.bmm(M_intermediate, v_i) # [b, n, d] @ [b, d, d] -> [b, n, d]
                    M_next_head_input = torch.einsum('bni,bij->bnj', M_intermediate, v_i)
                else: # randomV == 0 (Identity V)
                    M_next_head_input = M_intermediate

                # 7. Normalization (Acts as Post-LayerNorm in this structure)
                # Normalize the result before feeding to the next head (or as final output)
                M_current = torch.nn.functional.normalize(M_next_head_input, dim=2)

            # --- End Head Loop (Sequential Update) ---

            # Final state for this step is the M_current after the last head
            self.M = M_current

        # Return updated time
        new_time = current_time + eta # Step advances by eta

        return new_time


# Functions moved to runner.py:
# - do_single_beta
# - do_results
# Function moved to plot_utils.py:
# - plot_pickle
