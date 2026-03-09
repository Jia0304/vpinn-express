# VPINN Express

**From CAD to Physics in Seconds. No Mesh, Just Intelligence.**

无网格 · 可微分 · 实时仿真智能体

---

## What is this?

VPINN Express 是一款基于 **Variational PINN (变分物理信息神经网络)** 的云原生 CAE 仿真平台 MVP。
用 Monte-Carlo 采样 + 神经网络训练替代传统 FEM 的网格划分流程，实现秒级物理场求解。

当前 Demo：**2D 带孔板 (Plate with a Hole) 平面应力分析**。

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit + Plotly |
| Solver | PyTorch (CPU) — Adam + LBFGS |
| Physics | Variational form (最小势能原理) |
| Geometry | Monte-Carlo point cloud (mesh-free) |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开 `http://localhost:8501`，调参后点击 "启动仿真" 即可。

## Project Structure

```
├── .streamlit/config.toml   # Streamlit 服务配置
├── vpinn_core/              # 求解器核心包
│   ├── config.py            #   SimulationConfig 数据类
│   ├── geometry.py          #   无网格采样点生成
│   ├── physics.py           #   弹性矩阵 & 应变计算
│   └── solver.py            #   Adam+LBFGS 训练引擎
├── app.py                   # Streamlit 入口
└── requirements.txt         # 依赖 (CPU-only PyTorch)
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select repo → branch `main` → entry `app.py`
4. Deploy (first build ~5 min)

## License

MIT
