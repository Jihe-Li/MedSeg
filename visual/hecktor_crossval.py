import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import io
from visual.funcs.violin import draw_custom_violinplot


if __name__ == "__main__":

    dice_fold0 = io.load_csv_to_arr('outputs/seg_fold0/results/csv/Epoch200_Step103200/dice.csv')
    dice_fold1 = io.load_csv_to_arr('outputs/seg_fold1/results/csv/Epoch200_Step102800/dice.csv')
    dice_fold2 = io.load_csv_to_arr('outputs/seg_fold2/results/csv/Epoch200_Step104400/dice.csv')
    # dice_fold3 = io.load_csv_to_arr('outputs/seg_fold3/results/csv/Epoch200_Step103200/dice.csv')
    dice_fold4 = io.load_csv_to_arr('outputs/seg_fold4/results/csv/Epoch100_Step52400/dice.csv')

    data_dict = {
        "Fold 0": {
            'Dice': dice_fold0
        },
        "Fold 1": {
            'Dice': dice_fold1
        },
        "Fold 2": {
            'Dice': dice_fold2
        },
        "Fold 4": {
            'Dice': dice_fold4
        }
    }

    draw_custom_violinplot(
        data_dict=data_dict,
        save_path='hecktor_crossval.svg',
        font_size=13,
        bg_color="white",
        grid_color="#d9d9d9",
        x_label="Dice Metrics",
        y_label="Score",
    )
