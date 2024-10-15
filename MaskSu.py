import torch
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# 设置Detectron2配置
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

# 加载图像
image_path = "/Users/dean/Desktop/uploads/588cb120a2ac4104a231101b20831e8f.jpeg"
image = cv2.imread(image_path)
outputs = predictor(image)

# 获取预测的掩码和类别
instances = outputs["instances"]
masks = instances.pred_masks.cpu().numpy()
classes = instances.pred_classes.cpu().numpy()

# 假设类别 0 是人
for i, mask in enumerate(masks):
    if classes[i] == 0:
        part_mask = mask.astype("uint8") * 255
        part_name = f"person_part_{i}"
        cv2.imwrite(f"{part_name}_mask.png", part_mask)

# 可视化结果
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Result", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()