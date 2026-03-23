import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import io
from visual.funcs.violin import draw_custom_violinplot
from visual.funcs.boxplot import draw_custom_boxplot


if __name__ == "__main__":

    metric_name = 'dice'
    dice_fold0 = io.load_csv_to_arr(f'outputs/seg_fold0/results/csv/Epoch200_Step103200/{metric_name}.csv')
    dice_fold1 = io.load_csv_to_arr(f'outputs/seg_fold1/results/csv/Epoch200_Step102800/{metric_name}.csv')
    dice_fold2 = io.load_csv_to_arr(f'outputs/seg_fold2/results/csv/Epoch200_Step104400/{metric_name}.csv')
    dice_fold3 = io.load_csv_to_arr(f'outputs/seg_fold3/results/csv/Epoch200_Step104800/{metric_name}.csv')
    dice_fold4 = io.load_csv_to_arr(f'outputs/seg_fold4/results/csv/Epoch200_Step104800/{metric_name}.csv')

    data_dict = {
        "Fold 0": {
            'GTVp': dice_fold0
        },
        "Fold 1": {
            'GTVp': dice_fold1
        },
        "Fold 2": {
            'GTVp': dice_fold2
        },
        "Fold 3": {
            'GTVp': dice_fold3
        },
        "Fold 4": {
            'GTVp': dice_fold4
        }
    }

    draw_custom_boxplot(
        data_dict=data_dict,
        save_path=f'outputs/hecktor_crossval_{metric_name}.svg',
        font_size=13,
        bg_color="white",
        grid_color="#d9d9d9",
        x_label="",
        y_label=metric_name,
        show_outliers=False,
    )
