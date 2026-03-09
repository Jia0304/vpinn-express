import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .config import SimulationConfig
from .physics import elasticity_matrix_plane_stress, compute_strain
from .geometry import load_domain_data, generate_neumann_boundary

# === 网络定义 ===
class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, hidden_layers=5, hidden_neurons=64):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_neurons), nn.SiLU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.SiLU()])
        layers.append(nn.Linear(hidden_neurons, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class VPINNModel(nn.Module):
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.config = config
        self.net = MLP()
        self.uv_scale = 0.1

    def forward(self, xy):
        # 归一化
        x, y = xy[:, 0:1], xy[:, 1:2]
        x_hat = (x - self.config.x_c) / self.config.L_ref
        y_hat = (y - self.config.y_c) / self.config.L_ref
        inp = torch.cat([x_hat, y_hat], dim=1)
        
        raw_uv = self.net(inp)
        
        # 硬约束: y_bottom 处位移为 0
        # 防止除以0，分母加一个小量
        d_D = (y - self.config.y_bottom) / 100.0 
        return d_D * raw_uv * self.uv_scale

# === 主求解函数 ===
def run_solver(config: SimulationConfig):
    # 根据配置选择设备
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"[Core] Starting Solver on {device}...")
    
    # 1. 准备数据
    vol_xy_np, vol_w_np = load_domain_data(config)
    # 生成边界点 (此处使用默认参数，也可改为从config读取)
    bnd_xy_np, bnd_w_np = generate_neumann_boundary() 
    
    vol_xy = torch.from_numpy(vol_xy_np).float().to(device)
    vol_w = torch.from_numpy(vol_w_np).float().to(device)
    bnd_xy = torch.from_numpy(bnd_xy_np).float().to(device)
    bnd_w = torch.from_numpy(bnd_w_np).float().to(device)
    
    # 2. 初始化模型与物理矩阵
    model = VPINNModel(config).to(device)
    C_mat = elasticity_matrix_plane_stress(config.E, config.nu, device)
    
    # 3. 定义 Loss 函数 (闭包)
    def calculate_loss():
        # 必须开启梯度，因为计算应变需要求导
        vol_xy.requires_grad_(True)
        
        u = model(vol_xy)
        eps = compute_strain(u, vol_xy)
        sigma = torch.matmul(eps, C_mat.T)
        
        # 能量泛函 (应变能)
        energy = 0.5 * (eps * sigma).sum(dim=1)
        vol_term = (energy * vol_w).sum()
        
        # 外力做功 (Neumann)
        u_bnd = model(bnd_xy)
        work_term = (u_bnd[:, 1] * config.sigma0 * bnd_w).sum()
        
        return vol_term - work_term

    # 4. 训练循环 (Adam)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss_history = []
    
    print("[Core] Running Adam Optimizer...")
    for epoch in range(config.epochs):
        optimizer.zero_grad()
        loss = calculate_loss()
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss_val:.4e}")

    # 5. 精细调整 (LBFGS)
    print("[Core] Running LBFGS Refinement...")
    lbfgs = optim.LBFGS(model.parameters(), 
                        history_size=50, 
                        max_iter=1000, 
                        line_search_fn="strong_wolfe")
    
    def closure():
        lbfgs.zero_grad()
        l = calculate_loss()
        l.backward()
        return l
    
    lbfgs.step(closure)
    
    # 6. 结果后处理 (生成纯数据用于前端渲染)
    # === 关键修正开始 ===
    # 注意：这里不能使用 torch.no_grad()，因为 compute_strain 内部有 torch.autograd.grad
    
    # 重新确保输入坐标开启梯度追踪
    vol_xy.requires_grad_(True)
    
    # 前向计算最终结果
    u_final = model(vol_xy)
    eps_final = compute_strain(u_final, vol_xy)
    sigma_final = torch.matmul(eps_final, C_mat.T)
    
    # 转 Numpy (必须使用 .detach() 来切断梯度，否则报错)
    res_xy = vol_xy.detach().cpu().numpy()
    res_u = u_final.detach().cpu().numpy()
    res_s = sigma_final.detach().cpu().numpy() # [S11, S22, S12]
    
    # 计算 Von Mises 应力
    s11, s22, s12 = res_s[:, 0], res_s[:, 1], res_s[:, 2]
    von_mises = np.sqrt(s11**2 - s11*s22 + s22**2 + 3*s12**2)
    # === 关键修正结束 ===
    
    return {
        "status": "success",
        "loss_history": loss_history,
        "max_stress": float(np.max(von_mises)),
        "data": {
            "x": res_xy[:, 0].tolist(),
            "y": res_xy[:, 1].tolist(),
            "u_mag": np.linalg.norm(res_u, axis=1).tolist(),
            "von_mises": von_mises.tolist()
        }
    }