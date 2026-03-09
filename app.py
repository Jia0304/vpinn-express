# -*- coding: utf-8 -*-
"""
VPINN Express — Web Demo (Streamlit)
=====================================
无网格 · 可微分 · 实时仿真智能体

数据流:
  Sidebar 参数 → SimulationConfig → run_solver() → result dict → Plotly 渲染
  结果缓存在 st.session_state.result，防止 Streamlit rerun 时重复计算。
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import time

# ── 导入 VPINN 核心库 ──────────────────────────────────────────────
from vpinn_core.config import SimulationConfig
from vpinn_core.solver import run_solver
from vpinn_core.geometry import generate_plate_with_hole

# ====================================================================
# 页面全局配置
# ====================================================================
st.set_page_config(
    page_title="VPINN Express",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自定义 CSS：收窄顶部留白、美化按钮 ─────────────────────────────
st.markdown(
    """
<style>
    /* 缩小顶部空白 */
    .block-container { padding-top: 1.5rem; }
    /* 主标题样式 */
    h1 { letter-spacing: 0.02em; }
    /* 隐藏 Streamlit 默认 footer */
    footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ====================================================================
# 初始化 Session State（仅首次运行时执行）
# ====================================================================
if "result" not in st.session_state:
    st.session_state.result = None  # 求解结果缓存
if "run_params" not in st.session_state:
    st.session_state.run_params = None  # 上次运行的参数快照

# ====================================================================
# 侧边栏 — 控制面板
# ====================================================================
with st.sidebar:
    st.header("⚙️ 工况设置")
    st.caption("调节参数后点击底部按钮启动仿真")

    st.markdown("---")

    # ── 1. 几何参数 ────────────────────────────────────────────────
    st.subheader("1. 几何参数")
    r_hole = st.slider(
        "开孔半径 r (mm)",
        min_value=2.0,
        max_value=20.0,
        value=10.0,
        step=0.5,
        help="圆孔板中心孔的半径，板尺寸固定为 50×100 mm",
    )
    n_points = st.slider(
        "采样点数",
        min_value=500,
        max_value=5000,
        value=2000,
        step=250,
        help="域内 Monte-Carlo 积分点数量，越多精度越高但越慢",
    )

    # ── 2. 载荷参数 ────────────────────────────────────────────────
    st.subheader("2. 载荷条件")
    sigma0 = st.number_input(
        "顶部拉应力 σ₀ (MPa)",
        min_value=10.0,
        max_value=500.0,
        value=100.0,
        step=10.0,
        help="沿 Y 方向施加在板顶部的均匀拉伸载荷",
    )

    # ── 3. 求解器参数 ──────────────────────────────────────────────
    st.subheader("3. 求解器 (Agent)")
    epochs = st.slider(
        "Adam 训练步数",
        min_value=500,
        max_value=5000,
        value=1000,
        step=500,
        help="Adam 优化器迭代次数，之后自动追加 LBFGS 精修",
    )

    st.markdown("---")

    # ── 启动按钮 ───────────────────────────────────────────────────
    run_btn = st.button("🚀 启动仿真", type="primary", use_container_width=True)

    # ── 侧边栏底部信息 ─────────────────────────────────────────────
    st.markdown("---")
    st.caption("VPINN Express v0.1 · Mesh-Free · Differentiable")

# ====================================================================
# 标题区
# ====================================================================
st.title("🚀 VPINN Express")
st.markdown(
    "**无网格 · 可微分 · 实时仿真智能体**　｜　Plate with a Hole — 2D 平面应力分析"
)

# ====================================================================
# 主计算流程（仅在点击按钮时触发）
# ====================================================================
if run_btn:
    # 用 st.status 展示多阶段 Agent 工作流
    with st.status("🤖 Agent 工作流启动中...", expanded=True) as status:
        # ── 阶段 1: Geo-Agent 生成几何 ────────────────────────────
        st.write("**🔷 Geo-Agent:** 正在生成无网格采样点...")
        t0 = time.time()
        geom_data = generate_plate_with_hole(
            n_points=n_points,
            r_hole=r_hole,
        )
        t_geo = time.time() - t0
        st.write(f"　✅ 生成 {len(geom_data['xy'])} 个积分点　({t_geo:.2f}s)")

        # ── 阶段 2: Trainer-Agent 训练网络 ─────────────────────────
        st.write("**🔷 Trainer-Agent:** 正在初始化神经网络 (MLP 5×64)...")
        cfg = SimulationConfig(
            sigma0=sigma0,
            epochs=epochs,
            domain_data_numpy=geom_data,
            device="cpu",  # 云端部署强制 CPU
        )

        st.write(f"**🔷 Trainer-Agent:** Adam ×{epochs} → LBFGS 精修中...")
        t1 = time.time()
        result = run_solver(cfg)
        t_solve = time.time() - t1
        st.write(f"　✅ 训练完成　({t_solve:.1f}s)")

        # ── 阶段 3: Analyst-Agent 后处理 ──────────────────────────
        st.write("**🔷 Analyst-Agent:** 正在渲染应力场与生成报告...")
        time.sleep(0.2)  # 模拟后处理延迟
        st.write("　✅ 后处理完成")

        # 更新状态栏
        status.update(
            label=f"✅ 仿真完成！总耗时 {t_geo + t_solve:.1f}s",
            state="complete",
            expanded=False,
        )

    # ── 缓存结果到 session_state ──────────────────────────────────
    st.session_state.result = result
    st.session_state.run_params = {
        "r_hole": r_hole,
        "sigma0": sigma0,
        "epochs": epochs,
        "n_points": n_points,
    }

# ====================================================================
# 结果展示区（从 session_state 读取，rerun 不会丢失）
# ====================================================================
res = st.session_state.result

if res is not None:
    # ── 准备 DataFrame ─────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "x": res["data"]["x"],
            "y": res["data"]["y"],
            "von_mises": res["data"]["von_mises"],
            "u_mag": res["data"]["u_mag"],
        }
    )

    # 取回上次运行参数（用于理论值计算和圆孔绘制）
    params = st.session_state.run_params or {}
    cur_r = params.get("r_hole", r_hole)
    cur_sigma = params.get("sigma0", sigma0)

    # ── 布局：左列 (云图) ｜ 右列 (KPI + Loss + 位移) ──────────────
    col_left, col_right = st.columns([3, 2], gap="large")

    # ================================================================
    # 左列：Von Mises 应力云图
    # ================================================================
    with col_left:
        st.subheader("📊 Von Mises 应力云图")

        fig_stress = go.Figure(
            data=go.Scatter(
                x=df["x"],
                y=df["y"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=df["von_mises"],
                    colorscale="Jet",
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="σ_vm (MPa)", side="right"),
                        thickness=15,
                    ),
                    cmin=float(df["von_mises"].quantile(0.01)),
                    cmax=float(df["von_mises"].quantile(0.99)),
                ),
                text=[
                    f"({x:.1f}, {y:.1f})<br>σ_vm = {s:.1f} MPa"
                    for x, y, s in zip(df["x"], df["y"], df["von_mises"])
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )

        # 绘制圆孔轮廓
        theta = np.linspace(0, 2 * np.pi, 120)
        fig_stress.add_trace(
            go.Scatter(
                x=cur_r * np.cos(theta),
                y=cur_r * np.sin(theta),
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # 保持物理比例
        x_pad = 5
        fig_stress.update_layout(
            height=700,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(
                title="X (mm)",
                range=[df["x"].min() - x_pad, df["x"].max() + x_pad],
                scaleanchor="y",
                scaleratio=1,
                constrain="domain",
            ),
            yaxis=dict(
                title="Y (mm)",
                range=[df["y"].min() - x_pad, df["y"].max() + x_pad],
            ),
            template="plotly_white",
        )

        st.plotly_chart(fig_stress, use_container_width=True)

        # 数据下载
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📥 下载结果 CSV",
            data=csv_data,
            file_name="vpinn_result.csv",
            mime="text/csv",
        )

    # ================================================================
    # 右列：KPI + Loss + 位移场
    # ================================================================
    with col_right:
        st.subheader("📈 Agent 监控面板")

        # ── KPI 指标卡 ────────────────────────────────────────────
        theoretical = cur_sigma * 3.0  # Kirsch 解析解 Kt=3
        max_stress = res["max_stress"]
        rel_error = (max_stress - theoretical) / theoretical * 100

        k1, k2, k3 = st.columns(3)
        k1.metric("最大应力", f"{max_stress:.1f} MPa")
        k2.metric("理论值 (Kt=3)", f"{theoretical:.1f} MPa")
        k3.metric("相对偏差", f"{rel_error:+.1f}%")

        st.markdown("---")

        # ── Loss 收敛曲线 ─────────────────────────────────────────
        st.markdown("**Loss 收敛曲线 (能量泛函)**")
        loss_arr = np.array(res["loss_history"])

        fig_loss = go.Figure(
            data=go.Scatter(
                y=loss_arr,
                mode="lines",
                line=dict(color="#636EFA", width=1.5),
                hovertemplate="Epoch %{x}<br>Loss = %{y:.4e}<extra></extra>",
            )
        )
        fig_loss.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=10, b=30),
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white",
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        st.markdown("---")

        # ── 位移场云图 ────────────────────────────────────────────
        st.markdown("**位移幅值 (Displacement Magnitude)**")

        fig_disp = go.Figure(
            data=go.Scatter(
                x=df["x"],
                y=df["y"],
                mode="markers",
                marker=dict(
                    size=3,
                    color=df["u_mag"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="u (mm)", side="right"),
                        thickness=12,
                    ),
                ),
                text=[
                    f"({x:.1f}, {y:.1f})<br>u = {u:.4f} mm"
                    for x, y, u in zip(df["x"], df["y"], df["u_mag"])
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )
        fig_disp.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(title="X (mm)", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Y (mm)"),
            template="plotly_white",
        )
        st.plotly_chart(fig_disp, use_container_width=True)

        # ── 折叠统计面板 ─────────────────────────────────────────
        with st.expander("📋 详细统计"):
            st.markdown(f"""
| 指标 | 值 |
|---|---|
| 采样点数 | {len(df)} |
| 应力范围 | {df["von_mises"].min():.2f} ~ {df["von_mises"].max():.2f} MPa |
| 平均应力 | {df["von_mises"].mean():.2f} MPa |
| 位移范围 | {df["u_mag"].min():.5f} ~ {df["u_mag"].max():.5f} mm |
| 训练轮数 | {len(res["loss_history"])} (Adam) + LBFGS |
| 最终 Loss | {res["loss_history"][-1]:.4e} |
""")

# ====================================================================
# 占位符（尚未运行时的引导界面）
# ====================================================================
else:
    st.info("👈 请在左侧设置参数，然后点击 **启动仿真** 开始计算。")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
### 📝 使用说明
1. **调整孔径** — 拖动滑块设置圆孔半径 r
2. **设置载荷** — 输入顶部拉应力 σ₀
3. **启动仿真** — 点击按钮，Agent 自动完成几何→训练→后处理
4. **查看结果** — 左侧应力云图，右侧 KPI 与 Loss 曲线
""")
    with col_b:
        st.markdown("""
### 🔬 技术原理
- **无网格**: Monte-Carlo 采样替代 FEM 网格划分
- **变分形式**: 最小势能原理 (能量泛函) 替代强形式残差
- **自动微分**: PyTorch autograd 计算应变/应力张量
- **硬约束**: 底部位移 Dirichlet BC 直接编码进网络
- **双阶段优化**: Adam 全局搜索 + LBFGS 局部精修
""")
