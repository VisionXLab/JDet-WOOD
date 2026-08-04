"""Continue an older v2 run from a healthy epoch-6 checkpoint."""

_base_ = './point2rbox_v2_1x_dota.py'

name = 'point2rbox_v2_1x_dota_edge_buffer_fix'
work_dir = 'work_dirs/point2rbox_v2_1x_dota_edge_buffer_fix'
resume_path = 'work_dirs/point2rbox_v2_1x_dota/checkpoints/ckpt_6.pkl'
