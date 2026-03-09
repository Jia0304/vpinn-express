# VPINN Core Module
# 无网格可微分仿真核心库

from .config import SimulationConfig
from .solver import run_solver
from .geometry import (
    generate_plate_with_hole,
    load_domain_data,
    generate_neumann_boundary,
)
from .physics import elasticity_matrix_plane_stress, compute_strain

# geometry_stp 依赖 pythonocc-core，云端部署时可能不可用，做安全导入
try:
    from .geometry_stp import sample_from_step_file, sample_from_step_file_path
except ImportError:
    sample_from_step_file = None
    sample_from_step_file_path = None

__all__ = [
    "SimulationConfig",
    "run_solver",
    "generate_plate_with_hole",
    "load_domain_data",
    "generate_neumann_boundary",
    "elasticity_matrix_plane_stress",
    "compute_strain",
    "sample_from_step_file",
    "sample_from_step_file_path",
]
