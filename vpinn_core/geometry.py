import numpy as np
import io
from scipy.stats import qmc

def load_domain_data(config):
    """
    加载积分点数据 (Collocation Points)
    支持两种模式：
    1. 网页上传的 .npz 文件流 (BytesIO)
    2. 内存中直接传入的 Numpy 字典 (用于实时生成)
    """
    # 优先检查是否直接传入了 numpy 数据 (来自 Agent 实时生成)
    if config.domain_data_numpy is not None:
        data = config.domain_data_numpy
        return data["xy"], data["w"]
    
    # 其次检查是否有文件流 (来自网页上传)
    elif config.domain_data_bytes is not None:
        try:
            with io.BytesIO(config.domain_data_bytes) as f:
                data = np.load(f)
                return data["xy"], data["w"]
        except Exception as e:
            raise ValueError(f"无法解析上传的 npz 文件: {e}")
            
    else:
        raise ValueError("Config 中未提供 domain_data_bytes 或 domain_data_numpy")

def generate_neumann_boundary(n_points=2048, x_range=(-25.0, 25.0), y_top=50.0):
    """
    生成顶部受力边界点 (使用 Sobol 序列保证均匀性)
    n_points: 建议使用 2 的幂次方 (如 1024, 2048) 以避免 Scipy 警告
    """
    sampler = qmc.Sobol(d=1, scramble=True)
    sb = sampler.random(n_points)
    
    x_min, x_max = x_range
    xn = x_min + (x_max - x_min) * sb.flatten()
    yn = np.full_like(xn, y_top)
    
    # 边界积分权重 = 边长 / 点数
    length = x_max - x_min
    wn = np.full_like(xn, length / n_points)
    
    return np.stack([xn, yn], axis=1), wn

def generate_plate_with_hole(n_points=2000, width=50.0, height=100.0, r_hole=10.0):
    """
    [增强功能] 实时生成带孔圆板的采样点
    用于 VPINN Express 网页版，允许用户拖动滑块改变孔径
    
    参数:
        n_points: 期望的域内积分点数量
        width: 板宽 (x方向)
        height: 板高 (y方向)
        r_hole: 中间圆孔半径
    """
    # 1. 候选点生成 (Monte Carlo)
    # 为了保证挖去孔后剩下的点数够用，先多生成 50% 的点
    n_candidate = int(n_points * 1.5)
    
    # x: [-width/2, width/2], y: [-height/2, height/2]
    x = np.random.uniform(-width/2, width/2, n_candidate)
    y = np.random.uniform(-height/2, height/2, n_candidate)
    
    # 2. 挖孔逻辑 (几何过滤)
    # 计算每个点到原点的距离
    dist_sq = x**2 + y**2
    
    # 保留距离大于孔半径的点 (r^2)
    mask = dist_sq >= r_hole**2
    
    x_valid = x[mask]
    y_valid = y[mask]
    
    # 3. 截断或填充
    # 如果点不够，其实也不用报错，VPINN 对点数不敏感，但为了严谨我们截取前 n_points
    if len(x_valid) > n_points:
        x_valid = x_valid[:n_points]
        y_valid = y_valid[:n_points]
    
    real_n = len(x_valid)
    
    # 4. 计算积分权重 (Volume/Area Integration Weight)
    # 权重 w = 总面积 / 点数
    # 总面积 = 矩形面积 - 圆孔面积
    area_rect = width * height
    area_hole = np.pi * r_hole**2
    total_area = area_rect - area_hole
    
    if real_n > 0:
        weight_val = total_area / real_n
    else:
        weight_val = 0.0
        
    w_valid = np.full_like(x_valid, weight_val)
    
    # 返回字典，结构与 npz 文件保持一致，方便 solver.py 调用
    return {
        "xy": np.stack([x_valid, y_valid], axis=1), 
        "w": w_valid
    }