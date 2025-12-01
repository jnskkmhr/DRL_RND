import torch 

def get_cartpole_cbf_ocp_constraint(state, u_des)->tuple[torch.Tensor, torch.Tensor]:
    """
    define QP matrices for CBF-OCP for cartpole environment. 
    
    Here is OCP formulation:
    a* = argmin_a 0.5 * (a - u_des)^2
    s.t. Lf_h(x) + Lg_h(x) * a + alpha * h(x) >= 0
    
    args:
        state: (N, 4) torch tensor
        u_des: (N,) torch tensor
    returns:
        P: (N,) torch tensor
        q: (N,) torch tensor
        A: (N,) torch tensor
        b: (N,) torch tensor
    """
    # params
    M = 1.0 # cart mass
    m = 0.1 # pole mass
    l = 1.0 # pole length
    g = 9.81 # gravity
    
    x_bound = 0.7
    alpha_1 = 5
    alpha_2 = 100
    
    # unpack state
    p = state[:, 0]
    theta = state[:, 1]
    p_dot = state[:, 2]
    theta_dot = state[:, 3]
    
    # get system dynamics
    mass_mat = torch.zeros((state.shape[0], 2, 2))
    mass_mat[:, 0, 0] = M + m
    mass_mat[:, 0, 1] = m * l * torch.cos(theta)
    mass_mat[:, 1, 0] = mass_mat[:, 0, 1]
    mass_mat[:, 1, 1] = m * l ** 2
    mass_mat_inv = torch.linalg.inv(mass_mat)
    
    corriolis_mat = torch.zeros((state.shape[0], 2))
    corriolis_mat[:, 0] = -m * l * torch.sin(theta) * theta_dot**2
    corriolis_mat[:, 1] = m * g * l * torch.sin(theta)
    
    actuation_mat = torch.zeros((state.shape[0], 2))
    actuation_mat[:, 0] = 1.0
    
    f = torch.zeros((state.shape[0], 4))
    g = torch.zeros((state.shape[0], 4))
    f[:, 0] = p_dot
    f[:, 1] = theta_dot
    f[:, 2:] = -(mass_mat_inv @ corriolis_mat.unsqueeze(-1)).squeeze(-1)
    g[:, 2:] = (mass_mat_inv @ actuation_mat.unsqueeze(-1)).squeeze(-1)


    # get CBFs
    # h_1 = 0.5 * (x_bound - p) ** 2 # control input vanishes, thus we need higher order CBF h_2
    h_2 = -p*p_dot + 0.5*alpha_1*(x_bound**2 - p**2)
    
    
    dh_2_dx = torch.stack([-alpha_1*p-p_dot, torch.zeros_like(p), -p, torch.zeros_like(p)], dim=-1) # (N, 4)
    Lf_h_2 = (dh_2_dx * f).sum(dim=-1) # (N,)
    Lg_h_2 = (dh_2_dx * g).sum(dim=-1) # (N,)
    
    P = torch.ones(state.shape[0]).unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
    q = -2 * u_des # (N, 1)
    G = (-Lg_h_2).unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
    h = (Lf_h_2 + alpha_2 * h_2).unsqueeze(-1)  # (N, 1)
    
    return P, q, G, h
    
if __name__ == "__main__":
    # simple test
    states = torch.tensor([[0.5, 0.0, 0.0, 0.0],
                           [0.8, 0.0, 0.0, 0.0],
                           [1.2, 0.0, 0.0, 0.0]])
    u_des = torch.tensor([0.0, 0.0, 0.0])
    P, q, A, b = get_cartpole_cbf_ocp_constraint(states, u_des)
    print("P:", P)
    print("q:", q)
    print("A:", A)
    print("b:", b)