"""Final from-scratch v2 run with EdgeLoss and optimizer-resume fixes."""

_base_ = './point2rbox_v2_1x_dota.py'

name = 'point2rbox_v2_1x_dota_final_fixed'
work_dir = 'work_dirs/point2rbox_v2_1x_dota_final_fixed'
resume_path = None
