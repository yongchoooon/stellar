"""
Some configurations.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""
import numpy as np

language = 'ko'
do_filter = True
data_dir = '../../dataset-stellar/Syn_data_ko_filter/train/train-50k-ko-{i}/'
data_dir_eval = '../../dataset-stellar/Syn_data_ko_filter/eval/eval-1k-ko/'

# font
font_size = [25, 60]
underline_rate = 0.01
strong_rate = 0.05
oblique_rate = 0.02
font_dir = 'fonts/ko_ttf'
standard_font_path = 'fonts/ko_ttf/NanumGothic.ttf'

# text
text_filepath = 'data/texts_ko.txt'

# background
bg_filepath = 'data/label.txt'

## background augment
brightness_rate = 0.8
brightness_min = 0.7
brightness_max = 1.3
color_rate = 0.8
color_min = 0.7
color_max = 1.3
contrast_rate = 0.8
contrast_min = 0.7
contrast_max = 1.3

# curve
is_curve_rate = 0.05
curve_rate_param = [0.1, 0]

# perspective
rotate_param = [1, 0]
zoom_param = [0.1, 1]
shear_param = [2, 0]
perspect_param = [0.0005, 0]

# render

## surf augment
elastic_rate = 0.001
elastic_grid_size = 4
elastic_magnitude = 2

## colorize
padding_ud = [0, 10]
padding_lr = [0, 20]
is_border_rate = 0.02
is_shadow_rate = 0.02
shadow_angle_degree = [1, 3, 5, 7]
shadow_angle_param = [0.5, None]
shadow_shift_param = np.array([[0, 1, 3], [2, 7, 15]], dtype = np.float32)
shadow_opacity_param = [0.1, 0.5]
color_filepath = 'data/colors_new.cp'
use_random_color_rate = 0.5

## data directory
i_t_dir = 'i_t'
i_s_dir = 'i_s'
t_sk_dir = 't_sk'
t_t_dir = 't_t'
t_b_dir = 't_b'
t_f_dir = 't_f'
s_s_dir = 's_s'
mask_t_dir = 'mask_t'
mask_s_dir = 'mask_s'


# sample
sample_num_train = 50000
N = 4

sample_num_eval = 1000

# multiprocess
process_num = 16
data_capacity = 64
