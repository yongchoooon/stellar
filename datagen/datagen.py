"""
Generating data for SRNet
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import os
import cv2
from tqdm import tqdm
import argparse
from Synthtext.gen import multiprocess_datagen
import importlib.util
import types

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(cfg, language, data_dir, mode, do_filter, sample_num):

    i_t_dir = os.path.join(data_dir, cfg.i_t_dir)
    i_s_dir = os.path.join(data_dir, cfg.i_s_dir)
    t_sk_dir = os.path.join(data_dir, cfg.t_sk_dir)
    t_t_dir = os.path.join(data_dir, cfg.t_t_dir)
    t_b_dir = os.path.join(data_dir, cfg.t_b_dir)
    t_f_dir = os.path.join(data_dir, cfg.t_f_dir)
    s_s_dir = os.path.join(data_dir, cfg.s_s_dir)
    mask_t_dir = os.path.join(data_dir, cfg.mask_t_dir)
    mask_s_dir = os.path.join(data_dir, cfg.mask_s_dir)
    
    makedirs(i_t_dir)
    makedirs(i_s_dir)
    makedirs(t_sk_dir)
    makedirs(t_t_dir)
    makedirs(t_b_dir)
    makedirs(t_f_dir)
    makedirs(s_s_dir)
    makedirs(mask_t_dir)
    makedirs(mask_s_dir)

    it_txt = open(os.path.join(data_dir, 'i_t.txt'), 'w')
    is_txt = open(os.path.join(data_dir, 'i_s.txt'), 'w')
    font_txt = open(os.path.join(data_dir, 'font.txt'), 'w')

    mp_gen = multiprocess_datagen(
        cfg,
        cfg.process_num,
        cfg.data_capacity,
        language = language,
        mode = mode,
        do_filter = do_filter
    )
    
    mp_gen.multiprocess_runningqueue()
    digit_num = len(str(sample_num))
    for idx in tqdm(range(sample_num)):
        data_dict = mp_gen.dequeue_data()
        i_t, i_s, t_sk, t_t, t_b, t_f, s_s, mask_t, mask_s =data_dict["data"]
        is_text = data_dict["is_text"]
        it_text = data_dict["it_text"]
        font_name = data_dict["font"]
        is_txt.write(str(idx).zfill(digit_num) + '.png' + ' ' + is_text + '\n')
        it_txt.write(str(idx).zfill(digit_num) + '.png' + ' ' + it_text + '\n')
        font_txt.write(str(idx).zfill(digit_num) + '.png' + ' ' + font_name + '\n')
        i_t_path = os.path.join(i_t_dir, str(idx).zfill(digit_num) + '.png')
        i_s_path = os.path.join(i_s_dir, str(idx).zfill(digit_num) + '.png')
        t_sk_path = os.path.join(t_sk_dir, str(idx).zfill(digit_num) + '.png')
        t_t_path = os.path.join(t_t_dir, str(idx).zfill(digit_num) + '.png')
        t_b_path = os.path.join(t_b_dir, str(idx).zfill(digit_num) + '.png')
        t_f_path = os.path.join(t_f_dir, str(idx).zfill(digit_num) + '.png')
        s_s_path = os.path.join(s_s_dir, str(idx).zfill(digit_num) + '.png')
        mask_t_path = os.path.join(data_dir, cfg.mask_t_dir, str(idx).zfill(digit_num) + '.png')
        mask_s_path = os.path.join(data_dir, cfg.mask_s_dir, str(idx).zfill(digit_num) + '.png')
        cv2.imwrite(i_t_path, i_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(i_s_path, i_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_sk_path, t_sk, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_t_path, t_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_b_path, t_b, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(t_f_path, t_f, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(s_s_path, s_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(mask_t_path, mask_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(mask_s_path, mask_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    mp_gen.terminate_pool()
    is_txt.close()
    it_txt.close()
    font_txt.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Language")
    parser.add_argument("-c", "--config", type=str, required=True)

    args = parser.parse_args()
    config_path = args.config

    config_path = f'configs/{config_path}.py'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist.")
    module_name = os.path.splitext(os.path.basename(config_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    cfg_dict = {
        k: v for k, v in cfg.__dict__.items()
        if not k.startswith('__') and not callable(v) and not isinstance(v, types.ModuleType)
    }
    cfg = types.SimpleNamespace(**cfg_dict)

    language = cfg.language
    do_filter = cfg.do_filter


    print("=================================")
    print(f"do_filter : {do_filter}")
    print("=================================")

    for i in range(1, cfg.N + 1):
        mode = 'train'
        data_dir = cfg.data_dir.format(i = i)
        sample_num = cfg.sample_num_train
        main(cfg, language, data_dir, mode, do_filter, sample_num)
    
    mode = 'eval'
    data_dir = cfg.data_dir_eval
    sample_num = cfg.sample_num_eval
    main(cfg, language, data_dir, mode, do_filter, sample_num)