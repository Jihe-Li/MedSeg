import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def draw_custom_violinplot(data_dict, save_path, font_size=12, bg_color='white', grid_color='#e0e0e0', x_label='Metrics', y_label='Values'):
    """
    绘制分组小提琴图
    
    参数:
    data_dict (dict): 数据字典，格式如 {"Method A": {"Dice_GTVp": [...], ...}, ...}
    font_size (int): 全局字体大小
    bg_color (str): 图表背景颜色
    grid_color (str): 网格线颜色
    x_label (str): X轴标签名称
    y_label (str): Y轴标签名称
    """
    # 1. 数据转换：将嵌套字典转换为 DataFrame (长格式)
    records = []
    for method, metrics in data_dict.items():
        for metric, values in metrics.items():
            for val in values:
                records.append({
                    'Method': method,
                    'Metric': metric,
                    'Value': val
                })
    df = pd.DataFrame(records)
    
    # 2. 设置全局样式与字体
    plt.rcParams.update({'font.size': font_size})
    
    # 3. 创建画布并设置背景色
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # 4. 绘制横向网格线 (置于图层底层)
    ax.yaxis.grid(True, color=grid_color, linestyle='-', linewidth=1.0, alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True) # 确保网格线在图形下方
    
    # 5. 绘制小提琴图
    # x="Metric" 对应 Dice_*, hue="Method" 对应不同的算法/葫芦
    # split=False 表示不同的 Method 并排显示，如果只有两个 Method 且想拼在一起可以设为 True
    # inner="box" 会自动绘制你图中的中位数白点、四分位距粗线和晶须细线
    sns.violinplot(
        data=df, 
        x='Metric', 
        y='Value', 
        hue='Method',
        inner='box', 
        palette='Set2', # 可自定义颜色盘，例如 ['#86C6B9', '#F4A582']
        linewidth=1.5,
        cut=0,          # 不向数据范围外延伸
        bw_adjust=0.8,  # 可选：减小平滑导致的“鼓包”
        ax=ax
    )
    # ax.set_ylim(0, 1)   # Dice 显式限制到 [0,1]
    
    # 6. 完善标签和图例细节
    ax.set_xlabel(x_label, fontsize=font_size + 2)
    ax.set_ylabel(y_label, fontsize=font_size + 2)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    
    # 去除顶部和右侧的边框线使其更美观
    sns.despine()
    
    # 调整图例位置
    plt.legend(title='Methods', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    import numpy as np
    
    # 模拟生成一些正态分布的测试数据
    np.random.seed(42)
    sample_data = {
        "Method A": {
            "Dice_GTVp": np.random.normal(0.85, 0.05, 100),
            "Dice_GTVn": np.random.normal(0.75, 0.08, 100)
        },
        "Method B": {
            "Dice_GTVp": np.random.normal(0.88, 0.04, 100),
            "Dice_GTVn": np.random.normal(0.80, 0.06, 100)
        }
    }
    
    # 调用函数绘制
    draw_custom_violinplot(
        data_dict=sample_data,
        save_path='violin.svg',
        font_size=12,
        bg_color='#FAFAFA', # 浅灰白背景
        grid_color='#D3D3D3',
        x_label='Evaluation Metrics',
        y_label='Dice Score'
    )