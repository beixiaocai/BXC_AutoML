from paddleocr import PaddleOCR
import cv2
import time

# 初始化OCR引擎
ocr = PaddleOCR(
    det_model_dir='models/plate_det/inference_model/Student2',  # 车牌检测模型路径
    rec_model_dir='models/plate_rec/inference_model',  # 车牌识别模型路径
    use_angle_cls=False,                  # 车牌无需文字方向分类
    lang='ch',                            # 中文模型
    use_gpu=True                         # 启用GPU加速（如可用）
)

# 读取测试图片
image_path = 'data/1.jpg'
image_path = 'data/2.jpg'

img = cv2.imread(image_path)

t1 = time.time()
# 执行OCR
result = ocr.ocr(img, det=True, rec=True)

# 解析结果
for idx, plate in enumerate(result[0]):
    det_box = plate[0]  # 检测框坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    rec_text = plate[1][0]  # 识别文本
    rec_score = plate[1][1]  # 识别置信度

    print(f"车牌 {idx + 1}:")
    print(f"  位置: {det_box}")
    print(f"  识别结果: {rec_text} (置信度: {rec_score:.4f})")

    # 可视化结果（可选）
    for i in range(4):
        start = tuple(map(int, det_box[i]))
        end = tuple(map(int, det_box[(i + 1) % 4]))
        cv2.line(img, start, end, (0, 255, 0), 2)
    cv2.putText(img, rec_text, tuple(map(int, det_box[0])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# 保存可视化结果
# cv2.imwrite('result.jpg', img)
t2 = time.time()
print("infer 1 image, spend ", t2 - t1)