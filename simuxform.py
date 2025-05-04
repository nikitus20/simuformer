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
        self.last_regen_time_int = -1 # Track last integer time for regeneration

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

        # --- Use Precomputed Matrices FIRST if provided --- 
        if precomputed_BF is not None and precomputed_V is not None:
            # print(f'[DEBUG Head {i}] Using precomputed BF and V.') # Optional debug print
            # Ensure they are on the correct device
            self.BF[i] = precomputed_BF.to(device)
            self.V[i] = precomputed_V.to(device)
            return # Done for this head
        
        # --- If not precomputed, check for NOANNEAL=2 reuse --- 
        if args.noanneal == 2 and self.BF[i] is not None and self.V[i] is not None:
             print(f'[DEBUG Head {i}] noanneal=2, reusing existing BF and V.')
             return # Done for this head

        # --- If not precomputed and not reused, GENERATE matrices ---
        # Determine batch size for generation based on noanneal
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
            elif args.randomKQ == 4 or args.randomKQ == 5: # Modes 4 and 5 use same init
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
            elif args.randomV == 4 or args.randomV == 5: # Modes 4 and 5 use same init
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
            # Aggregate V*softmax(K^T Q)
            # M shape [batch, ntokens, dmodel]
            # BF[i] shape [batch?, dmodel, dmodel]
            # V[i] shape [batch?, dmodel, dmodel]

            # Need to handle potential batch dim in BF/V if noanneal=0
            # Let's ensure BF/V are expanded for batch if needed
            # M2 is the aggregation accumulator, re-initialize
            self.M2.zero_() 

            for i in range(n_heads):
                # Handle potential batch dimension mismatch (pp vs batch)
                # Ensure bf_i and v_i have a batch dimension before expanding
                # bf_i_base = self.BF[i].unsqueeze(0) if self.BF[i].dim() == 2 else self.BF[i]
                # v_i_base = self.V[i].unsqueeze(0) if self.V[i].dim() == 2 else self.V[i]
                # bf_i = bf_i_base.expand(batch, -1, -1) if bf_i_base.size(0) == 1 and batch > 1 else bf_i_base
                # v_i = v_i_base.expand(batch, -1, -1) if v_i_base.size(0) == 1 and batch > 1 else v_i_base
                
                # Simpler approach: directly unsqueeze and expand if needed.
                bf_i = self.BF[i]
                if bf_i.dim() == 2:
                     bf_i = bf_i.unsqueeze(0).expand(batch, -1, -1)
                elif bf_i.size(0) == 1 and batch > 1:
                     bf_i = bf_i.expand(batch, -1, -1)
                # Ensure bf_i is now [batch, dmodel, dmodel]

                v_i = self.V[i]
                if v_i.dim() == 2:
                     v_i = v_i.unsqueeze(0).expand(batch, -1, -1)
                elif v_i.size(0) == 1 and batch > 1:
                     v_i = v_i.expand(batch, -1, -1)
                 # Ensure v_i is now [batch, dmodel, dmodel]

                # Calculate Q^T K equivalent: M @ BF[i] @ M^T
                # Use torch.baddbmm for potentially better performance/memory
                # Input (M): [b, n, d], Mat1 (BF[i]): [b, d, d], Mat2 (M.transpose): [b, d, n]
                # Result (SM): [b, n, n]
                self.SM = torch.bmm(torch.bmm(self.M, bf_i), self.M.transpose(1, 2))

                # Apply causal masking if sequential
                if args.sequential:
                     mask = torch.triu(torch.full((ntokens, ntokens), float('-inf'), device=device), diagonal=1)
                     self.SM += mask.unsqueeze(0) # Add mask with batch dim

                # Apply beta scaling
                self.SM *= self.beta

                # Normalize attention scores
                use_softmax = getattr(args, 'use_softmax', True) 
                if use_softmax:
                     self.SM = torch.softmax(self.SM, dim=2)
                else: # Use scaled exp without full softmax normalization
                     self.SM = torch.exp(self.SM)

                # Calculate aggregated values: A = SM @ M @ V[i]
                # Result (A_head): [b, n, d]
                # Use torch.baddbmm or chained bmm
                A_head = torch.bmm(self.SM, torch.bmm(self.M, v_i))

                # Accumulate attention output scaled by 1/n_heads
                self.M2 += A_head / n_heads 
            
            # M2 now holds the aggregated attention output 'A' from the paper's notation
            A = self.M2 

            # Apply normalization and update steps based on the mode
            if args.norm == 'postln':
                # Simple Euler step: M_next = M + eta * A
                # Then normalize M_next
                self.M += eta * A
                self.M = torch.nn.functional.normalize(self.M, dim=2)
                # r remains 1 (or isn't used)
                
            elif args.norm == 'preln':
                # Update r: r_dot = <r, A> => r_next = r + eta * <r, A>
                # Project A onto the tangent space of M: A_tangent = A - <M, A> * M
                # Update M: M_next = M + eta * A_tangent / r 
                # Then normalize M_next (residual connection)

                # <r, A> calculation (sum over dmodel dimension)
                rA_dot = torch.sum(self.r.unsqueeze(-1) * A, dim=-1)
                self.r += eta * rA_dot

                # <M, A> projection term
                MA_proj = torch.sum(self.M * A, dim=-1, keepdim=True) * self.M
                A_tangent = A - MA_proj

                # Update M using r in the denominator (add epsilon for stability)
                self.M += eta * A_tangent / (self.r.unsqueeze(-1) + 1e-9)
                self.M = torch.nn.functional.normalize(self.M, dim=2)

            elif args.norm == 'periln':
                 # Calculate A_norm = |A| + eps
                 A_norm = torch.norm(A, dim=-1, keepdim=True) + 1e-9
                 A_normalized = A / A_norm
                 
                 # Update r: r_dot = <r, A/|A|> => r_next = r + eta * <r, A/|A|>
                 rA_norm_dot = torch.sum(self.r.unsqueeze(-1) * A_normalized, dim=-1)
                 self.r += eta * rA_norm_dot

                 # Update M: M_dot = A / (r * |A|) => M_next = M + eta * A / (r * |A|)
                 # Note: A / (r * |A|) = (A / |A|) / r = A_normalized / r
                 # Project M_dot onto tangent space? No, the update eq is just M_dot = A/(r|A|)
                 # Then normalize M_next 
                 self.M += eta * A_normalized / (self.r.unsqueeze(-1) + 1e-9)
                 self.M = torch.nn.functional.normalize(self.M, dim=2)

            elif args.norm == 'ngpt':
                 # r_k = 1 (implicitly, self.r is not updated or used in the M update)
                 # M_dot = 0.2 * A / |A| => M_next = M + eta * 0.2 * A / |A|
                 # Then normalize M_next

                 # Calculate A_norm = |A| + eps
                 A_norm = torch.norm(A, dim=-1, keepdim=True) + 1e-9
                 A_normalized = A / A_norm

                 self.M += eta * 0.2 * A_normalized
                 self.M = torch.nn.functional.normalize(self.M, dim=2)

            else:
                raise ValueError(f"Unknown normalization mode: {args.norm}")

        # --- Regeneration Logic for modes 5 ---
        new_time = current_time + eta 
        new_time_int = math.floor(new_time)
        
        if new_time_int > self.last_regen_time_int and (args.randomKQ == 5 or args.randomV == 5):
            print(f"[INFO] Regenerating KQ/V matrices at time t ~ {new_time:.2f} (crossed integer step {new_time_int})", file=sys.stderr)
            for i in range(self.n_heads):
                 # Call generate_KQV without precomputed args to force regeneration
                 # It will respect noanneal internally
                 self.generate_KQV(i) 
            self.last_regen_time_int = new_time_int

        # Return updated time
        return new_time

