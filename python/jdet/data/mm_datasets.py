# DOTA-txt family dataset loaders.
#
# 类别元组逐字取自 /root/ref/Point2RBox-v3/mmrotate/datasets/ 各文件的
# METAINFO['classes']（顺序即 label id，不得重排）。这些数据集的标注均为
# DOTA txt 格式（每行 8 坐标 + 类名 [+ difficulty]），加载逻辑完全复用
# P2RV2DOTADataset：
#   - 未知类名跳过（对齐 star.py L130-131 / 基类 L97-98）
#   - difficulty > diff_thr 跳过（对齐各 ref 的 diff_thr 语义；
#     RSAR/OCDPCB 的 txt 若无 difficulty 列按 0 处理）
#   - 图像后缀由 IMG_SUFFIX 给出；文件不存在时按 RSAR ref 的扩展名列表回退
#
# XML/COCO/json datasets use their dedicated adapters in this package.

from jdet.utils.registry import DATASETS
from .p2rv2_dota import P2RV2DOTADataset


@DATASETS.register_module()
class DOTAv15Dataset(P2RV2DOTADataset):
    """DOTA-v1.5（dotav15.py METAINFO，16 类）。"""
    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter',
               'container-crane')
    IMG_SUFFIX = '.png'


@DATASETS.register_module()
class DOTAv2Dataset(P2RV2DOTADataset):
    """DOTA-v2.0（dotav2.py METAINFO，18 类）。"""
    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter',
               'container-crane', 'airport', 'helipad')
    IMG_SUFFIX = '.png'


@DATASETS.register_module()
class STARDataset(P2RV2DOTADataset):
    """STAR（star.py METAINFO，48 类；ref img_suffix 默认 'png'）。"""
    CLASSES = ('ship', 'boat', 'crane', 'goods_yard', 'tank', 'storehouse',
               'breakwater', 'dock', 'airplane', 'boarding_bridge', 'runway',
               'taxiway', 'terminal', 'apron', 'gas_station', 'truck', 'car',
               'truck_parking', 'car_parking', 'bridge', 'cooling_tower',
               'chimney', 'vapor', 'smoke', 'genset', 'coal_yard',
               'lattice_tower', 'substation', 'wind_mill',
               'cement_concrete_pavement', 'toll_gate', 'flood_dam',
               'gravity_dam', 'ship_lock', 'ground_track_field',
               'basketball_court', 'engineering_vehicle', 'foundation_pit',
               'intersection', 'soccer_ball_field', 'tennis_court',
               'tower_crane', 'unfinished_building', 'arch_dam', 'roundabout',
               'baseball_diamond', 'stadium', 'containment_vessel')
    IMG_SUFFIX = '.png'


@DATASETS.register_module()
class RSARDataset(P2RV2DOTADataset):
    """RSAR（rsar.py METAINFO，6 类；图像扩展名不统一，靠基类回退解析）。"""
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')
    IMG_SUFFIX = '.png'


@DATASETS.register_module()
class OCDPCBDataset(P2RV2DOTADataset):
    """OCD-PCB（ocdpcb.py METAINFO，41 类；ref 固定 .png）。"""
    CLASSES = ('C', 'J', 'RS', 'CE', 'IC-SOT23', 'IC-SOP', 'IC-TO252',
               'IC-SOT223', 'D', 'JW', 'X', 'R', 'IC-BGA', 'IC-QFN', 'SW',
               'SW-S', 'IC-SOT235', 'IC-SOT89', 'IC-QFP', 'IC-SOT234', 'LED',
               'IC-SON', 'CA', 'LR', 'IC-SOT236', 'JN-FFC', 'RN-N', 'JN-XHH',
               'CN', 'RN', 'JN-DF', 'JN-DM', 'JN-XHV', 'IC', 'P', 'DC', 'LA',
               'LB', 'X-HC49', 'JN', 'F')
    IMG_SUFFIX = '.png'
