import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# 配置Detectron2模型
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置阈值
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# 加载图像
image = cv2.imread('path_to_image.jpg')

# 使用Detectron2预测掩码
outputs = predictor(image)
masks = outputs["instances"].pred_masks.to("cpu").numpy()

# 初始化一个空白掩码图像，与原图像大小相同
height, width = image.shape[:2]
combined_mask = np.zeros((height, width), dtype=np.uint8)

# 遍历所有检测到的目标掩码
for i in range(masks.shape[0]):
    # 提取第i个掩码，并转换为标准二值掩码格式
    mask = masks[i].astype(np.uint8) * 255

    # 可以选择将单独的掩码保存或进一步处理
    cv2.imwrite(f'mask_{i}.png', mask)

    # 合并所有掩码，生成一个总的掩码
    combined_mask = np.maximum(combined_mask, mask)

# 保存总的掩码
cv2.imwrite('combined_mask.png', combined_mask)