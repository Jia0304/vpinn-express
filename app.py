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
from vpinn_core.geometry_stp import sample_from_step_file

# ====================================================================
# 页面全局配置
# ====================================================================
st.set_page_config(
    page_title="VPINN Express",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { letter-spacing: 0.02em; }
    footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ====================================================================
# 初始化 Session State
# ====================================================================
if "result" not in st.session_state:
    st.session_state.result = None
if "run_params" not in st.session_state:
    st.session_state.run_params = None
if "geom_data" not in st.session_state:
    st.session_state.geom_data = None  # STP/NPZ 预处理后的点云
if "geom_bounds" not in st.session_state:
    st.session_state.geom_bounds = None  # STP 解析出的几何边界信息

# ====================================================================
# 侧边栏 — 控制面板
# ====================================================================
with st.sidebar:
    st.header("⚙️ 工况设置")
    st.caption("调节参数后点击底部按钮启动仿真")

    st.markdown("---")

    # ── 0. 几何输入方式 ────────────────────────────────────────────
    st.subheader("0. 几何输入方式")
    geom_mode = st.radio(
        "选择输入方式",
        ["⚙️ 参数化生成（圆孔板）", "📁 上传文件（STP / NPZ）"],
        index=0,
        help="参数化模式可拖动滑块调整孔径；上传模式支持 STEP 和 NPZ 文件",
    )

    st.markdown("---")

    # ── 1. 几何参数（根据模式切换） ────────────────────────────────
    st.subheader("1. 几何设置")

    if geom_mode == "⚙️ 参数化生成（圆孔板）":
        # ---- 参数化模式 ----
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
            help="域内 Monte-Carlo 积分点数量",
        )
        # 清空上传数据
        st.session_state.geom_data = None
        st.session_state.geom_bounds = None

    else:
        # ---- 上传模式 ----
        uploaded_file = st.file_uploader(
            "上传几何文件",
            type=["stp", "step", "npz"],
            help="支持 STEP (.stp/.step) 和采样点 (.npz) 格式",
        )

        n_points = st.slider(
            "采样点数（STP 模式生效）",
            min_value=500,
            max_value=5000,
            value=2000,
            step=250,
        )

        # 预处理按钮
        if st.button("🔍 预处理几何", use_container_width=True):
            if uploaded_file is None:
                st.warning("请先上传文件")
            else:
                fname = uploaded_file.name.lower()
                with st.spinner("正在解析文件..."):
                    try:
                        raw_bytes = uploaded_file.read()
                        uploaded_file.seek(0)  # 重置指针

                        if fname.endswith((".stp", ".step")):
                            # ── STP 解析 ──
                            geom_result = sample_from_step_file(
                                raw_bytes,
                                n_points=n_points,
                            )
                            st.session_state.geom_data = {
                                "xy": geom_result["xy"],
                                "w": geom_result["w"],
                            }
                            st.session_state.geom_bounds = geom_result["bounds"]

                        elif fname.endswith(".npz"):
                            # ── NPZ 直接加载 ──
                            data = np.load(uploaded_file)
                            st.session_state.geom_data = {
                                "xy": data["xy"],
                                "w": data["w"],
                            }
                            # 从点云推断边界
                            xy = data["xy"]
                            st.session_state.geom_bounds = {
                                "x_min": float(xy[:, 0].min()),
                                "x_max": float(xy[:, 0].max()),
                                "y_min": float(xy[:, 1].min()),
                                "y_max": float(xy[:, 1].max()),
                                "hole_radius": 10.0,  # NPZ 无法自动检测
                            }

                        n_loaded = len(st.session_state.geom_data["xy"])
                        st.success(f"✅ 采样完成！{n_loaded} 个点")

                    except Exception as e:
                        st.error(f"文件处理失败: {e}")
                        st.session_state.geom_data = None
                        st.session_state.geom_bounds = None

        # 显示已解析的几何信息
        if st.session_state.geom_bounds is not None:
            b = st.session_state.geom_bounds
            st.markdown("**几何信息**")
            st.text(
                f"X: [{b['x_min']:.1f}, {b['x_max']:.1f}]\n"
                f"Y: [{b['y_min']:.1f}, {b['y_max']:.1f}]\n"
                f"孔径: {b['hole_radius']:.1f} mm"
            )

        # 上传模式下 r_hole 从解析结果取
        r_hole = (
            st.session_state.geom_bounds["hole_radius"]
            if st.session_state.geom_bounds
            else 10.0
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
    # 上传模式下，未预处理时禁用按钮
    if geom_mode == "📁 上传文件（STP / NPZ）" and st.session_state.geom_data is None:
        st.warning("请先上传文件并点击「预处理几何」")
        run_btn = st.button(
            "🚀 启动仿真",
            type="primary",
            use_container_width=True,
            disabled=True,
        )
    else:
        run_btn = st.button(
            "🚀 启动仿真",
            type="primary",
            use_container_width=True,
        )

    st.markdown("---")
    st.caption("VPINN Express v0.2 · Mesh-Free · Differentiable")

# ====================================================================
# 标题区
# ====================================================================
st.title("🚀 VPINN Express")
st.markdown(
    "**无网格 · 可微分 · 实时仿真智能体**　｜　Plate with a Hole — 2D 平面应力分析"
)

# ====================================================================
# 主计算流程
# ====================================================================
if run_btn:
    with st.status("🤖 Agent 工作流启动中...", expanded=True) as status:
        # ── 阶段 1: Geo-Agent ─────────────────────────────────────
        if geom_mode == "⚙️ 参数化生成（圆孔板）":
            st.write("**🔷 Geo-Agent:** 正在生成无网格采样点...")
            t0 = time.time()
            geom_data = generate_plate_with_hole(
                n_points=n_points,
                r_hole=r_hole,
            )
            t_geo = time.time() - t0
            st.write(f"　✅ 生成 {len(geom_data['xy'])} 个积分点　({t_geo:.2f}s)")
        else:
            st.write("**🔷 Geo-Agent:** 正在加载已预处理的采样点...")
            t0 = time.time()
            geom_data = st.session_state.geom_data
            t_geo = time.time() - t0
            st.write(f"　✅ 加载 {len(geom_data['xy'])} 个积分点　({t_geo:.2f}s)")

        # ── 阶段 2: Trainer-Agent ─────────────────────────────────
        st.write("**🔷 Trainer-Agent:** 正在初始化神经网络 (MLP 5×64)...")
        cfg = SimulationConfig(
            sigma0=sigma0,
            epochs=epochs,
            domain_data_numpy=geom_data,
            device="cpu",
        )

        st.write(f"**🔷 Trainer-Agent:** Adam ×{epochs} → LBFGS 精修中...")
        t1 = time.time()
        result = run_solver(cfg)
        t_solve = time.time() - t1
        st.write(f"　✅ 训练完成　({t_solve:.1f}s)")

        # ── 阶段 3: Analyst-Agent ─────────────────────────────────
        st.write("**🔷 Analyst-Agent:** 正在渲染应力场与生成报告...")
        time.sleep(0.2)
        st.write("　✅ 后处理完成")

        status.update(
            label=f"✅ 仿真完成！总耗时 {t_geo + t_solve:.1f}s",
            state="complete",
            expanded=False,
        )

    st.session_state.result = result
    st.session_state.run_params = {
        "r_hole": r_hole,
        "sigma0": sigma0,
        "epochs": epochs,
        "n_points": n_points,
    }

# ====================================================================
# 结果展示区
# ====================================================================
res = st.session_state.result

if res is not None:
    df = pd.DataFrame(
        {
            "x": res["data"]["x"],
            "y": res["data"]["y"],
            "von_mises": res["data"]["von_mises"],
            "u_mag": res["data"]["u_mag"],
        }
    )

    params = st.session_state.run_params or {}
    cur_r = params.get("r_hole", r_hole)
    cur_sigma = params.get("sigma0", sigma0)

    col_left, col_right = st.columns([3, 2], gap="large")

    # ── 左列：应力云图 ─────────────────────────────────────────────
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
                        title=dict(text="σ_vm (MPa)", side="right"), thickness=15
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
                title="Y (mm)", range=[df["y"].min() - x_pad, df["y"].max() + x_pad]
            ),
            template="plotly_white",
        )
        st.plotly_chart(fig_stress, use_container_width=True)

        csv_data = df.to_csv(index=False)
        st.download_button(
            "📥 下载结果 CSV",
            data=csv_data,
            file_name="vpinn_result.csv",
            mime="text/csv",
        )

    # ── 右列：KPI + Loss + 位移 ───────────────────────────────────
    with col_right:
        st.subheader("📈 Agent 监控面板")

        theoretical = cur_sigma * 3.0
        max_stress = res["max_stress"]
        rel_error = (max_stress - theoretical) / theoretical * 100

        k1, k2, k3 = st.columns(3)
        k1.metric("最大应力", f"{max_stress:.1f} MPa")
        k2.metric("理论值 (Kt=3)", f"{theoretical:.1f} MPa")
        k3.metric("相对偏差", f"{rel_error:+.1f}%")

        st.markdown("---")

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
                        title=dict(text="u (mm)", side="right"), thickness=12
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
# 占位引导界面
# ====================================================================
else:
    st.info("👈 请在左侧设置参数，然后点击 **启动仿真** 开始计算。")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
### 📝 使用说明
**模式 A — 参数化生成**
1. 选择「参数化生成」，拖动滑块设置孔径
2. 设置载荷 → 启动仿真

**模式 B — 上传文件**
1. 选择「上传文件」，上传 `.stp` 或 `.npz`
2. 点击「预处理几何」解析采样点
3. 设置载荷 → 启动仿真
""")
    with col_b:
        st.markdown("""
### 🔬 技术原理
- **无网格**: Monte-Carlo 采样替代 FEM 网格划分
- **变分形式**: 最小势能原理 (能量泛函) 替代强形式残差
- **自动微分**: PyTorch autograd 计算应变/应力张量
- **硬约束**: 底部位移 Dirichlet BC 直接编码进网络
- **双阶段优化**: Adam 全局搜索 + LBFGS 局部精修

### 📂 支持的文件格式
| 格式 | 说明 |
|---|---|
| `.stp` / `.step` | STEP AP214/AP203，自动解析边界与圆孔 |
| `.npz` | NumPy 压缩包，需含 `xy` 和 `w` 数组 |
""")
