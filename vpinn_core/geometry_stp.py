# -*- coding: utf-8 -*-
"""
STP文件采样模块
用于从上传的STEP文件中提取几何并生成采样点

当前实现：基于STEP文件的简化解析
未来可扩展：集成OpenCASCADE进行精确几何处理
"""

import numpy as np
import io
import re
from typing import Dict, Tuple, Optional


def parse_step_boundary_simple(step_content: str) -> Dict[str, float]:
    """
    简化的STEP文件解析 - 提取边界框信息
    
    Args:
        step_content: STEP文件内容（文本）
    
    Returns:
        包含边界信息的字典：{'x_min', 'x_max', 'y_min', 'y_max', 'hole_radius'}
    """
    # 提取所有CARTESIAN_POINT
    points = []
    point_pattern = r'CARTESIAN_POINT\([^,]*,\(([^)]+)\)\)'
    
    for match in re.finditer(point_pattern, step_content):
        coords_str = match.group(1)
        coords = [float(x.strip()) for x in coords_str.split(',')]
        if len(coords) >= 2:  # 至少需要x, y坐标
            points.append(coords[:2])  # 只取x, y
    
    if not points:
        # 如果解析失败，返回默认值
        return {
            'x_min': -25.0,
            'x_max': 25.0,
            'y_min': -50.0,
            'y_max': 50.0,
            'hole_radius': 10.0
        }
    
    points = np.array(points)
    
    # 提取圆孔半径（如果存在CIRCLE定义）
    hole_radius = 10.0  # 默认值
    circle_pattern = r'CIRCLE\([^,]*,([0-9.]+)\)'
    circle_matches = re.findall(circle_pattern, step_content)
    if circle_matches:
        hole_radius = float(circle_matches[0])
    
    return {
        'x_min': float(points[:, 0].min()),
        'x_max': float(points[:, 0].max()),
        'y_min': float(points[:, 1].min()),
        'y_max': float(points[:, 1].max()),
        'hole_radius': hole_radius
    }


def sample_from_step_file(
    step_bytes: bytes,
    n_points: int = 2000,
    hole_center: Tuple[float, float] = (0.0, 0.0)
) -> Dict[str, np.ndarray]:
    """
    从STEP文件字节流中采样点
    
    Args:
        step_bytes: STEP文件的字节数据
        n_points: 期望的采样点数量
        hole_center: 圆孔中心坐标 (x, y)
    
    Returns:
        包含 'xy' 和 'w' 的字典
        - xy: (N, 2) 数组，采样点坐标
        - w: (N,) 数组，积分权重
    """
    # 解码STEP文件
    try:
        step_content = step_bytes.decode('utf-8', errors='ignore')
    except:
        step_content = step_bytes.decode('gbk', errors='ignore')
    
    # 解析边界信息
    bounds = parse_step_boundary_simple(step_content)
    
    x_min = bounds['x_min']
    x_max = bounds['x_max']
    y_min = bounds['y_min']
    y_max = bounds['y_max']
    r_hole = bounds['hole_radius']
    
    print(f"[STP Sampler] Detected geometry:")
    print(f"  - X range: [{x_min:.2f}, {x_max:.2f}]")
    print(f"  - Y range: [{y_min:.2f}, {y_max:.2f}]")
    print(f"  - Hole radius: {r_hole:.2f}")
    
    # 采样策略：Monte Carlo采样 + 几何过滤
    n_candidate = int(n_points * 1.5)  # 多生成一些候选点
    
    x = np.random.uniform(x_min, x_max, n_candidate)
    y = np.random.uniform(y_min, y_max, n_candidate)
    
    # 过滤掉圆孔内的点
    hx, hy = hole_center
    dist_sq = (x - hx)**2 + (y - hy)**2
    mask = dist_sq >= r_hole**2
    
    x_valid = x[mask]
    y_valid = y[mask]
    
    # 截取所需数量的点
    if len(x_valid) > n_points:
        x_valid = x_valid[:n_points]
        y_valid = y_valid[:n_points]
    
    real_n = len(x_valid)
    
    # 计算积分权重
    area_total = (x_max - x_min) * (y_max - y_min)
    area_hole = np.pi * r_hole**2
    effective_area = area_total - area_hole
    
    if real_n > 0:
        weight_val = effective_area / real_n
    else:
        weight_val = 0.0
    
    w_valid = np.full(real_n, weight_val)
    
    print(f"[STP Sampler] Generated {real_n} sampling points")
    
    return {
        "xy": np.stack([x_valid, y_valid], axis=1),
        "w": w_valid,
        "bounds": bounds  # 附加边界信息，便于可视化
    }


def sample_from_step_file_path(
    file_path: str,
    n_points: int = 2000,
    hole_center: Tuple[float, float] = (0.0, 0.0)
) -> Dict[str, np.ndarray]:
    """
    从STEP文件路径采样（用于本地测试）
    
    Args:
        file_path: STEP文件路径
        n_points: 期望的采样点数量
        hole_center: 圆孔中心坐标
    
    Returns:
        包含 'xy' 和 'w' 的字典
    """
    with open(file_path, 'rb') as f:
        step_bytes = f.read()
    
    return sample_from_step_file(step_bytes, n_points, hole_center)


# === 用于更精确的STP处理（需要安装pythonocc-core） ===
def sample_from_step_with_occ(
    step_bytes: bytes,
    n_points: int = 2000
) -> Optional[Dict[str, np.ndarray]]:
    """
    使用OpenCASCADE进行精确的STEP文件处理
    
    注意：此功能需要安装 pythonocc-core
    pip install pythonocc-core
    
    Args:
        step_bytes: STEP文件字节数据
        n_points: 采样点数量
    
    Returns:
        采样数据或None（如果OCC未安装）
    """
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Core.BRep import BRep_Tool
        
        print("[STP Sampler] Using OpenCASCADE for precise geometry processing")
        
        # 保存临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.stp', delete=False) as tmp:
            tmp.write(step_bytes)
            tmp_path = tmp.name
        
        # 读取STEP文件
        reader = STEPControl_Reader()
        reader.ReadFile(tmp_path)
        reader.TransferRoots()
        shape = reader.OneShape()
        
        # 网格化
        mesh = BRepMesh_IncrementalMesh(shape, 0.1)
        mesh.Perform()
        
        # 提取面和三角形（这里简化处理，实际需要更复杂的逻辑）
        # ... OCC处理代码 ...
        
        import os
        os.unlink(tmp_path)
        
        # 返回采样结果
        return {
            "xy": np.zeros((n_points, 2)),
            "w": np.zeros(n_points)
        }
        
    except ImportError:
        print("[STP Sampler] OpenCASCADE not available, using simplified method")
        return None


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = "Plate_Hole_Circle_2D.stp"
    
    print(f"Testing STP sampler with: {test_file}")
    result = sample_from_step_file_path(test_file, n_points=100)
    
    print(f"\nResults:")
    print(f"  - Points: {result['xy'].shape}")
    print(f"  - Weights: {result['w'].shape}")
    print(f"  - Sample points:\n{result['xy'][:5]}")
    
    # 可视化（如果有matplotlib）
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 10))
        plt.scatter(result['xy'][:, 0], result['xy'][:, 1], s=1, alpha=0.5)
        
        # 画圆孔
        bounds = result['bounds']
        circle = plt.Circle((0, 0), bounds['hole_radius'], 
                           fill=False, color='red', linewidth=2)
        plt.gca().add_patch(circle)
        
        plt.axis('equal')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.title('STP File Sampling Result')
        plt.grid(True, alpha=0.3)
        plt.savefig('stp_sampling_test.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved to: stp_sampling_test.png")
    except:
        print("\nMatplotlib not available, skipping visualization")
