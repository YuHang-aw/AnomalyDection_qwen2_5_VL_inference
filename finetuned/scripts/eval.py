# 伪代码骨架：
# 1) 遍历 val 的 images；
# 2) 调用已训练模型推理，得到 per_region 的 decision/score；
# 3) 与正样本 id 对齐，算 Precision/Recall/F1；
# 4) 图像级：任一ROI异常→判为异常，算 P/R/F1/AUROC。
