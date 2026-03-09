from dataclasses import dataclass
from typing import Optional

@dataclass
class SimulationConfig:
    """仿真配置对象 (Data Transfer Object)"""
    # 物理参数
    E: float = 210000.0
    nu: float = 0.3
    sigma0: float = 100.0  # 远场拉应力
    
    # 几何参数 (用于归一化)
    x_c: float = 0.0
    y_c: float = 0.0
    L_ref: float = 100.0
    y_bottom: float = -50.0 # 用于硬约束
    
    # 训练参数
    epochs: int = 5000
    lr: float = 1e-3
    device: str = "cuda"
    
    # 输入数据 (二进制流)
    # 在网页版中，这里将接收上传的 npz 文件内容
    domain_data_bytes: Optional[bytes] = None 
    # 或者直接传 numpy 数组 (本地调试用)
    domain_data_numpy: Optional[dict] = None