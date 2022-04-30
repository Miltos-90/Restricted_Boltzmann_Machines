import torch
from torch.nn import functional as F

class RBM(torch.nn.Module):
    
    def __init__(self, hidden_units, visible_units, learning_rate, momentum = 0, weight_decay = 0, 
                 sparse_cost = 0, sparse_decay_rate = 0, sparse_probability = 0, seed = 123):
        ''' Initialisation function 
            Inputs [all scalars]:
                hidden units:       No. hidden units in the RBM 
                visible_units:      No. visible units in the RBM 
                learning_rate :     Learning rate for SGD
                momentum:           Momentum for SGD
                weight_decay:       Weight decay for SGD
                sparse_cost:        Sparsity penalty cost
                sparse_decay_rate:  Decay rate for sparsity penalty
                sparse_probability: Required probability for sparse activations
                seed:               Seed for weight initialisation
        '''
        
        super().__init__()
        
        self.lr       = learning_rate      # Learning rate for SGD
        self.m        = momentum           # Momentum for SGD
        self.wd       = weight_decay       # Weight decay
        self.C_sp     = sparse_cost        # Sparsity penalty cost
        self.wd_sp    = sparse_decay_rate  # Sparsity penalty decay rate
        self.p_req_sp = sparse_probability # Required activation probability of hidden units
        
        self.M = visible_units
        self.N = hidden_units
        
        # Initialise Weight matrix
        self.W = torch.empty(hidden_units, visible_units)
        torch.manual_seed(seed) # Reproducibility
        torch.nn.init.normal_(self.W, mean = 0, std = 0.1)
        self.c = torch.zeros(1, hidden_units) # Initialise Bias vector for hidden units
        self.b = torch.zeros(1, visible_units) # Initialise Bias vector for visible units
        
        # Internal variables for SGD (w/ momentum)
        self.ph_pre = 0
        self.DW_pre = 0
        self.Db_pre = 0
        self.Dc_pre = 0
    
    
    def ContrastiveDivergence(self, v, k):
        ''' k-step Contrastive Divergence training algorithm
            Inputs:
                v: batch [tensor (batch size x visible units)]
                k: No. steps for Gibbs sampling [scalar]
            Outputs:
                error: Reconstuction error [scalar]
        '''
        
        # Run Gibbs sampling
        p_h_v_pos, p_h_v_neg, _, _, v_neg = self.GibbsSampler(v, k)

        # Update parameters
        self.step(p_h_v_pos, p_h_v_neg, v, v_neg)

        # Compute reconstruction error 
        error = self.reconstruction_error(v, v_neg)
        
        return error
    
    def ParallelTempering(self, X, Nt, k):
        ''' Parallel tempering training algorithm 
            Inputs:
                X:  batch [tensor (batch size x visible units)]
                Nt: No. temperatures for PT [scalar]
                k:  No. steps for Gibbs sampling [scalar]
            Outputs:
                error: Reconstruction error [scalar]
        '''
        
        # Make temperature vector (these are inverse temperatures)
        T = torch.linspace(start = 0, end = 1, steps = Nt)
        
        # Make Gibbs chains and swap them
        V, H = self.make_Gibbs_chains(T, X, k)
        V, H = self.swap_Gibbs_chain(T = T, V = V, H = H, T0 = Nt - 2)
        V, H = self.swap_Gibbs_chain(T = T, V = V, H = H, T0 = Nt - 1)

        # Get final matrix
        v_neg = V[-1,:,:]

        # Compute probability vectors
        p_h_v_pos = RBM.p(X, self.W, self.c)
        p_h_v_neg = RBM.p(v_neg, self.W, self.c)

        # Update parameters
        self.step(p_h_v_pos, p_h_v_neg, X, v_neg)

        # Compute reconstruction error
        error = RBM.reconstruction_error(X, v_neg)
        
        return error

    
    def GibbsSampler(self, v_pos, k):
        ''' k-step Gibbs sampler
            Inputs:
                v_pos: batch [tensor (batch size x visible units)]
                k:     No. steps for Gibbs sampling [scalar]
            Outputs:
                p_h_v_pos: Probability of hidden unit being active (Positive phase) [tensor (batch size x hidden units)]
                p_h_v_neg: Probability of hidden unit being active (Negative phase) [tensor (batch size x hidden units)]
                p_v_h_neg: Probability of visible unit being active (Positive phase) [tensor (batch size x visible units)]
                h_pos:     Hidden states (Positive phase) [tensor (batch size x hidden units)]
                v_neg:     Visible states (Negative phase [tensor (btch size x visible units)]
        '''
        
        for _ in range(k):

            # Positive phase
            p_h_v_pos = RBM.p(v_pos, self.W, self.c)
            h_pos     = RBM.sample(p_h_v_pos)

            # Negative phase
            p_v_h_neg = RBM.p(h_pos, self.W.t(), self.b)
            v_neg     = RBM.sample(p_v_h_neg)
            p_h_v_neg = RBM.p(v_neg, self.W, self.c)
    
        return p_h_v_pos, p_h_v_neg, p_v_h_neg, h_pos, v_neg
    
    
    def sparsisty_penalty(self, p_h_active):
        '''Computation of sparsity penalty 
            Inputs: 
                p_h_active: Average probability of hidden unit being active [tensor (batch size x 1)]
            Outputs:
                sparsity_penalty: Penalty [tensor (batch size x 1)]
        '''
        
        if self.C_sp > 0:
        
            # Exponentially decaying average
            p_h_active = self.wd_sp * self.ph_pre + (1 - self.wd_sp) * p_h_active

            # Sparsity penalty
            sparsity_penalty = self.C_sp * (p_h_active - self.p_req_sp)
            
        else:
            sparsity_penalty = torch.zeros(self.c.shape[1])
            
        return sparsity_penalty
    
    
    def step(self, p_h_v_pos, p_h_v_neg, v_pos, v_neg):
        ''' Parameter updates using SGD w/ momentum 
            Inputs:
                p_h_v_pos: Probability of hidden unit being active (Positive phase) [tensor (batch size x hidden units)]
                p_h_v_neg: Probability of hidden unit being active (Negative phase) [tensor (batch size x hidden units)]
                p_v_h_neg: Probability of visible unit being active (Positive phase) [tensor (batch size x visible units)]
                v_neg:     Visible states (Negative phase [tensor (btch size x visible units)]
                v_pos:     Visible states (Negative phase [tensor (btch size x visible units)]
            Outputs: None
        '''
        
        # Actual probability of a hidden unit being active
        p_h_active = torch.mean(p_h_v_pos, dim = 0)
    
        # Compute sparsity penalty
        sparse_penalty = self.sparsisty_penalty(p_h_active)

        # Compute gradients
        dW = RBM.p_HvV(p_h_v_pos, v_pos) - RBM.p_HvV(p_h_v_neg, v_neg)
        db = (v_pos - v_neg).mean(dim = 0)
        dc = (p_h_v_pos - p_h_v_neg).mean(dim = 0)

        # Compute parameter updates
        Delta_w = self.m * self.DW_pre + self.lr * dW - self.wd * self.W - sparse_penalty[:, None]
        Delta_b = self.m * self.Db_pre + self.lr * db
        Delta_c = self.m * self.Dc_pre + self.lr * dc - sparse_penalty

        # Step
        self.W += Delta_w
        self.b += Delta_b
        self.c += Delta_c
        
        # Update for next iteration
        self.ph_pre = p_h_active
        self.DW_pre = Delta_w
        self.Db_pre = Delta_b
        self.Dc_pre = Delta_c
    
        return
    
    
    def free_energy(self, v, h, w = None):
        ''' Computation of free energy 
            E = - sum_{i in visible} b_i * v_i 
                - sum_{j in hidden} c_j * h_i 
                - sum_{i in visible, j in hidden} v_i, h_j, w_ij
                
            Inputs:
                v: Visible states (Negative phase) [tensor (batch size x visible units)]
                h: Hidden states (Negative phase) [tensor (batch size x visible units)]
                w: RBM weights [tensor (hidden units x visible units)]
            Outputs:
                E: Free energy [tensor (batch size x 1)]
        '''

        if w is None: w = self.W
        
        E = - (torch.matmul(v, self.b.t()) + torch.matmul(h, self.c.t())).squeeze() - (v * torch.matmul(h, w)).sum(dim = 1)

        return E
    
    
    def make_Gibbs_chains(self, T, X, k):
        ''' Generation of the multiple Gibbs chains for parallel tempering 
            Inputs:
                T: Temperature vector for parallel tempering [tensor (batch size x 1)]
                X: batch [tensor (batch size x visible units)]
                k: No. steps for Gibbs sampling [scalar]
            Outputs:
                V: Visible states (Negative phase) [tensor (No. temperatures x batch size x visible units)]
                H: Hidden states (Positive phase) [tensor (No. temperatures x batch size x hidden units)]
        '''
    
        # Batch size
        B = X.shape[0]
        
        # No. temperatures
        Nt = T.shape[0]

        # Initialise empty tensor to hold samples
        V = torch.empty(size  = (Nt, B, self.M))
        H = torch.empty(size  = (Nt, B, self.N))

        # Store RBM weights
        W0 = self.W.clone()

        # Loop over temperatures
        for idx in reversed(range(Nt)):

            # Set its parameters
            self.W = W0 * T[idx]

            # Run Gibbs sampler
            _, _, _, h_pos, v_neg = self.GibbsSampler(X, k)

            # Add to tensors
            V[idx, :, :] = v_neg
            H[idx, :, :] = h_pos

        # Restore weights
        self.W = W0

        return V, H
    
    
    def swap_Gibbs_chain(self, T, V, H, T0):
        ''' Swaps samples from two Gibbs chains
            Inputs:
                T:  Temperature vector for parallel tempering [tensor (batch size x 1)]
                V:  Visible states (Negative phase) [tensor (No. temperatures x batch size x visible units)]
                H:  Hidden states (Positive phase) [tensor (No. temperatures x batch size x hidden units)]
                T0: Index of T from which swapping starts [scalar]
            Outputs:
                V: Swapped isible states (Negative phase) [tensor (No. temperatures x batch size x visible units)]
                H: Swapped hidden states (Positive phase) [tensor (No. temperatures x batch size x hidden units)]
        '''
        
        for idx in range(T0, 1, -2):

            idx0 = idx - 1 # Index of previous temperature

            # Compute free energies
            E1 = self.free_energy(w = self.W * T[idx],  v = V[idx, :, :],  h = H[idx, :, :])
            E0 = self.free_energy(w = self.W * T[idx0], v = V[idx0, :, :], h = H[idx0, :, :])

            # Compute samples to be swapped
            p_swap   = RBM.swapping_probability(T[idx], T[idx0], E1, E0)
            swap_idx = RBM.to_swap(p_swap)

            # Swap matrices
            v_temp, h_temp = V[idx, swap_idx, :], H[idx, swap_idx, :]

            V[idx, swap_idx, :] = V[idx0, swap_idx, :]
            H[idx, swap_idx, :] = H[idx0, swap_idx, :]

            V[idx0, swap_idx, :] = v_temp
            H[idx0, swap_idx, :] = h_temp
        
        return V, H
        
    
    @staticmethod
    def p(X, W, bias):
        ''' Computation of p(h|v) or p(v|h) 
            Inputs:
                X: Batch [tensor (batch size x visible units) or (batch size x hidden units)]
                W: RBM Weights [tensor (hidden units x visible units)]
            Outputs: 
                p(h|v) or p(v|h) [tensor (batch size x visible units) or (batch size x hidden units)]
        '''
        return torch.sigmoid(F.linear(X, W, bias))

    
    @staticmethod
    def p_HvV(p, v):
        ''' Computation of p(H|v) x v 
            Inputs:
                v: Visible states (Negative phase) [tensor (batch size x visible units)]
                p: Probability of hidden unit being active (Negative phase) [tensor (batch size x hidden units)]
            Outputs: p(H|v) x v [tensor (visible units x visible units)]
        
        '''
        return torch.matmul(p.t(), v)
    
    
    @staticmethod
    def sample(p): 
        ''' Bernoulli sampling given probability p 
            Inputs:
                p: Probability [tensor (hidden units x 1) or (visible units x 1)]
            Outputs:
                Bernoulli sample [tensor (same as p)]
        '''
        return torch.bernoulli(p)
    
    
    @staticmethod
    def reconstruction_error(v_pos, v_neg):
        ''' Computation of reconstruction error 
            Inputs:
                v_pos: Visible units (Positive phase) [tensor (batch size x visible units)]
                v_neg: Visible units (Negative phase) [tensor (batch size x visible units)]
            Outputs:
                error: Average Reconstruction error [scalar]
        '''
        return torch.sum( (v_neg - v_pos) ** 2, dim = 1).mean().cpu().numpy()
    
    
    @staticmethod
    def swapping_probability(T1, T0, E1, E0):
        ''' Computation of swapping probability for parallel tempering 
            Inputs:
                T1, T0: Temperatures [scalars]
                E1, E0: Free energies [tensors (batch size x 1)]
            Outputs:
                Swapping probability [tensor (batch size x 1)]
        '''

        return torch.min(torch.ones_like(E1), torch.exp( (T1 - T0) * (E1 - E0) ) )

    
    @staticmethod
    def to_swap(p):
        ''' Boolean value with probability p for parallel tempering 
            Inputs:
                p: Swapping probability [tensor (batch size x 1)]
            Outputs:
                binary indices to swap [tensor (batch size x 1)]
        '''
        return torch.rand_like(p) >= 1 - p