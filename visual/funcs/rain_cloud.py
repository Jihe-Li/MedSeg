import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb


def draw_raincloud_plot(
    data_dict,
    font_size=12,
    bg_color="white",
    grid_color="#e0e0e0",
    x_label="Metrics",
    y_label="Value",
    figsize=(12, 7),
    palette=None,
    random_seed=42,
    show=True,
):
    """
    绘制 Raincloud 图（半小提琴 + 箱线图 + 抖动散点）。

    参数:
    data_dict (dict):
        嵌套字典，示例:
        {
            "Method A": {"Dice_GTVp": data, "Dice_GTVn": data},
            "Method B": {"Dice_GTVp": data, "Dice_GTVn": data}
        }
        其中:
        - "Dice_*" 为 x 轴分组
        - "Method *" 为同组内不同 plot
    font_size (int): 字体大小
    bg_color (str): 背景颜色
    grid_color (str): 网格线颜色
    x_label (str): x 轴名称
    y_label (str): y 轴名称
    figsize (tuple): 画布大小
    palette (list[str] | None): 方法颜色列表，长度不足时自动循环
    random_seed (int): 抖动散点随机种子
    show (bool): 是否立即 plt.show()
    """
    if not data_dict or not isinstance(data_dict, dict):
        raise ValueError("data_dict 必须是非空字典。")

    methods = list(data_dict.keys())
    if not methods:
        raise ValueError("未检测到方法（Method）数据。")

    metrics = []
    for method in methods:
        metric_dict = data_dict.get(method, {})
        if not isinstance(metric_dict, dict):
            raise ValueError(f"{method} 对应的数据必须是字典。")
        for metric in metric_dict.keys():
            if metric not in metrics:
                metrics.append(metric)

    if not metrics:
        raise ValueError("未检测到指标（Dice_*）数据。")

    def _to_clean_array(values):
        arr = np.asarray(values, dtype=float).ravel()
        return arr[np.isfinite(arr)]

    def _darken(color, factor=0.72):
        r, g, b = to_rgb(color)
        return (r * factor, g * factor, b * factor)

    def _draw_outer_violin_edge(ax_obj, verts, center_x, edge_color):
        # 只绘制右半侧外轮廓曲线，避免中心竖线出现
        mask = verts[:, 0] > center_x + 1e-6
        right_side = verts[mask]
        if right_side.shape[0] < 5:
            return
        order = np.argsort(right_side[:, 1])
        right_side = right_side[order]
        ax_obj.plot(
            right_side[:, 0],
            right_side[:, 1],
            color=edge_color,
            linewidth=1.8,
            alpha=0.95,
            zorder=2.5,
        )

    clean_data = {}
    for method in methods:
        clean_data[method] = {}
        for metric in metrics:
            values = data_dict[method].get(metric, [])
            clean_data[method][metric] = _to_clean_array(values)

    if palette is None:
        cmap = plt.get_cmap("tab10")
        palette = [cmap(i % 10) for i in range(len(methods))]
    elif len(palette) < len(methods):
        palette = [palette[i % len(palette)] for i in range(len(methods))]

    plt.rcParams.update({"font.size": font_size})
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    ax.yaxis.grid(True, color=grid_color, linestyle="-", linewidth=1.0, alpha=0.75)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    rng = np.random.default_rng(random_seed)
    metric_positions = np.arange(len(metrics), dtype=float)

    # 每个 metric 分配一个组宽；组内按方法均匀平移
    group_width = 0.8
    n_methods = len(methods)
    if n_methods == 1:
        method_offsets = [0.0]
    else:
        method_offsets = np.linspace(-group_width / 2, group_width / 2, n_methods)

    violin_width = min(0.26, group_width / max(n_methods, 1) * 0.9)
    scatter_size = 16

    for m_idx, method in enumerate(methods):
        color = palette[m_idx]
        for x_idx, metric in enumerate(metrics):
            vals = clean_data[method][metric]
            if vals.size == 0:
                continue

            center = metric_positions[x_idx] + method_offsets[m_idx]

            # 1) 半小提琴（保留右半侧）
            vp = ax.violinplot(
                dataset=[vals],
                positions=[center],
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in vp["bodies"]:
                verts = body.get_paths()[0].vertices
                verts[:, 0] = np.clip(verts[:, 0], center, np.inf)
                body.set_facecolor(color)
                body.set_edgecolor("none")
                body.set_alpha(0.28)
                body.set_linewidth(0.0)
                _draw_outer_violin_edge(ax, verts, center, color)

            # 2) 箱线图（放在中心略偏左，形成 raincloud 视觉）
            box_center = center - violin_width * 0.12
            bp = ax.boxplot(
                [vals],
                positions=[box_center],
                widths=violin_width * 0.32,
                patch_artist=True,
                showfliers=True,
                whis=1.5,
                manage_ticks=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_edgecolor(color)
                patch.set_alpha(0.45)
                patch.set_hatch("///")
                patch.set_linewidth(1.8)
            for whisk in bp["whiskers"]:
                whisk.set_color(color)
                whisk.set_linewidth(1.6)
            for cap in bp["caps"]:
                cap.set_color(color)
                cap.set_linewidth(1.6)
            for med in bp["medians"]:
                med.set_color(color)
                med.set_linewidth(2.0)
            for flier in bp["fliers"]:
                flier.set_marker("o")
                flier.set_markersize(5)
                flier.set_markerfacecolor("red")
                flier.set_markeredgecolor("red")
                flier.set_alpha(0.95)

            # 3) 抖动散点（放在中心右侧）
            jitter = rng.uniform(0.01, violin_width * 0.42, size=vals.size)
            x_points = np.full(vals.size, center) + jitter
            y_range = np.max(vals) - np.min(vals) if vals.size > 1 else 1.0
            shadow_dx = violin_width * 0.018
            shadow_dy = y_range * 0.008
            highlight_dx = -violin_width * 0.012
            highlight_dy = y_range * 0.005

            # 阴影层：模拟示例图中小球阴影
            ax.scatter(
                x_points + shadow_dx,
                vals - shadow_dy,
                s=scatter_size * 1.12,
                c=[(0, 0, 0)],
                alpha=0.18,
                linewidths=0,
                zorder=2.8,
            )

            # 主球层
            ax.scatter(
                x_points,
                vals,
                s=scatter_size * 0.98,
                c=[color],
                alpha=0.78,
                edgecolors=_darken(color, 0.62),
                linewidths=0.75,
                zorder=3,
            )

            # 高光层：增强“球体”质感
            ax.scatter(
                x_points + highlight_dx,
                vals + highlight_dy,
                s=scatter_size * 0.22,
                c="white",
                alpha=0.65,
                linewidths=0,
                zorder=3.2,
            )

    ax.set_xticks(metric_positions)
    ax.set_xticklabels(metrics, fontsize=font_size + 2)
    ax.set_xlabel(x_label, fontsize=font_size + 3)
    ax.set_ylabel(y_label, fontsize=font_size + 3)
    ax.tick_params(axis="y", labelsize=font_size + 1)

    legend_handles = [
        Patch(facecolor=palette[i], edgecolor=palette[i], alpha=0.45, label=methods[i])
        for i in range(len(methods))
    ]
    ax.legend(
        handles=legend_handles,
        title="Methods",
        fontsize=font_size,
        title_fontsize=font_size,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if show:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    # 简单示例
    demo = {
        "Method A": {
            "Dice_GTVp": np.random.normal(0.82, 0.06, 80),
            "Dice_GTVn": np.random.normal(0.74, 0.08, 80),
            "Dice_All": np.random.normal(0.78, 0.07, 80),
        },
        "Method B": {
            "Dice_GTVp": np.random.normal(0.86, 0.05, 80),
            "Dice_GTVn": np.random.normal(0.79, 0.07, 80),
            "Dice_All": np.random.normal(0.81, 0.06, 80),
        },
    }

    draw_raincloud_plot(
        data_dict=demo,
        font_size=13,
        bg_color="white",
        grid_color="#d9d9d9",
        x_label="Dice Metrics",
        y_label="Score",
    )
