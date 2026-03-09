import torch

def elasticity_matrix_plane_stress(E: float, nu: float, device: torch.device):
    """生成平面应力下的弹性矩阵 C"""
    coef = E / (1.0 - nu**2)
    C = torch.tensor([
        [1.0, nu, 0.0], 
        [nu, 1.0, 0.0], 
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ], dtype=torch.float32, device=device)
    return coef * C

def compute_strain(u: torch.Tensor, xy: torch.Tensor):
    """计算应变 (自动微分)"""
    # u shape: [N, 2], xy shape: [N, 2]
    # grad returns (dL/dx, dL/dy)
    grads = [torch.autograd.grad(u[:, i].sum(), xy, create_graph=True)[0] for i in range(2)]
    
    # eps_xx = du/dx (grads[0][:,0])
    # eps_yy = dv/dy (grads[1][:,1])
    # gam_xy = du/dy + dv/dx
    eps = torch.cat([
        grads[0][:, 0:1], 
        grads[1][:, 1:2], 
        grads[0][:, 1:2] + grads[1][:, 0:1]
    ], dim=1)
    return eps