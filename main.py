import argparse
import os
import time
from pathlib import Path
import cv2
import torch
from tqdm import tqdm
from engine import VisionMemoryEngine
from engine.utils import ensure_dir, list_images, parse_float_tuple2, parse_tuple2, select_roi_with_tk


def build_parser():
    parser = argparse.ArgumentParser(description="基于记忆库的视觉异常检测工具")

    # 运行模式说明：
    # train：使用正常样本构建记忆库
    # detect：对单张图片做异常检测
    # detect_batch：对目录中的所有图片做批量检测
    # calibrate_threshold：用一批正常图像估计推荐阈值
    # append_positive：向已有记忆库追加新的正样本特征
    parser.add_argument(
        "mode",
        choices=["train", "detect", "detect_batch", "calibrate_threshold", "append_positive"],
        help="运行模式：train / detect / detect_batch / calibrate_threshold / append_positive",
    )

    # 特征提取和检索所使用的设备，默认优先使用 CUDA。
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="运行设备，例如 cpu 或 cuda")

    # 主干网络，目前实现中仅支持 resnet50。
    parser.add_argument("--backbone", type=str, default="resnet50", help="特征提取主干网络，当前仅支持 resnet50")

    # 每个裁剪块送入主干网络前会被缩放到该尺寸。
    parser.add_argument("--input_size", type=int, nargs=2, default=[640, 640], metavar=("H", "W"), help="主干网络输入尺寸，高和宽，例如 --input_size 240 240")

    # 滑窗裁剪大小，越大上下文越多，但空间定位会更粗。
    parser.add_argument("--crop_size", type=int, nargs=2, default=[640, 640], metavar=("H", "W"), help="滑窗裁剪尺寸，高和宽，例如 --crop_size 160 160")

    # 滑窗步长；如果不填，内部默认使用 crop_size 的一半。
    parser.add_argument("--stride", type=int, nargs=2, default=[512, 512], metavar=("H", "W"), help="滑窗步长，高和宽；不填时默认使用 crop_size 的一半")

    # 训练时每次并行处理多少个裁剪块。
    parser.add_argument("--batch_size", type=int, default=32, help="训练构建记忆库时的批大小")

    # 检测和阈值校准时每次并行处理多少个裁剪块。
    parser.add_argument("--detect_batch_size", type=int, default=8, help="检测或阈值校准时的批大小")

    # 合并特征图后使用的局部平滑卷积核大小。
    parser.add_argument("--local_kernel", type=int, default=3, help="特征图局部平滑卷积核大小")

    # 构建最终记忆库前的压缩比例，越小保留的代表性特征越少。
    parser.add_argument("--memory_ratio", type=float, default=0.002, help="记忆库压缩比例，越小越省内存")

    # 随机投影后的目标特征维度。
    parser.add_argument("--target_embed_dimension", type=int, default=1024, help="随机投影后的目标特征维度")

    # 每个 patch 评分时使用的近邻个数。
    parser.add_argument("--knn_neighbors", type=int, default=1, help="KNN 近邻数量")

    # 近邻检索后端，auto 会在 CUDA 上优先选 torch，否则选 sklearn。
    parser.add_argument("--knn_backend", type=str, default="auto", choices=["auto", "torch", "sklearn", "faiss", "bm"], help="KNN 后端：auto、torch、sklearn、faiss 或 bm")

    # KNN 分块查询大小，用于控制显存或内存占用。
    parser.add_argument("--knn_query_chunk_size", type=int, default=8192, help="KNN 分块查询大小，用于控制内存占用")

    # BM1684 KNN 后端使用的 bmodel 路径。该 bmodel 应接收 [Q, D] 和 [N, D] 两个输入，并输出 [Q, N] 相似度矩阵。
    parser.add_argument("--bm_bmodel_path", type=str, default=None, help="BM1684 KNN bmodel 路径，启用 --knn_backend bm 时必填")

    # BM1684 设备号。
    parser.add_argument("--bm_device_id", type=int, default=0, help="BM1684 设备号")

    # BM1684 上数据库分块大小，用于控制单次矩阵乘规模。
    parser.add_argument("--bm_db_chunk_size", type=int, default=4096, help="BM1684 KNN 数据库分块大小")

    # 以下参数用于适配不同 bmodel 的 graph / 输入输出名字；不填时默认取第一个 graph 和默认的前两个输入、第一路输出。
    parser.add_argument("--bm_graph_name", type=str, default=None, help="BM1684 KNN graph 名称")
    parser.add_argument("--bm_query_input_name", type=str, default=None, help="BM1684 KNN 查询输入名")
    parser.add_argument("--bm_database_input_name", type=str, default=None, help="BM1684 KNN 数据库输入名")
    parser.add_argument("--bm_output_name", type=str, default=None, help="BM1684 KNN 输出名")

    # 在 CUDA 路径上启用混合精度。
    parser.add_argument("--use_amp", action="store_true", help="启用混合精度推理，仅在 CUDA 下生效")

    # 若大于 0，则先把图像长边缩放到不超过该值，再进入后续流程。
    parser.add_argument("--infer_long_side", type=int, default=0, help="推理前限制图像长边大小，0 表示不限制")

    # 训练用正常样本目录。
    parser.add_argument("--train_dir", type=str, help="训练模式下的正常样本目录")

    # 训练结束后保存模型的路径。
    parser.add_argument("--save_model", type=str, default="memory_model.pt", help="训练完成后保存模型的路径")

    # 检测、阈值校准、追加正样本时加载的模型路径。
    parser.add_argument("--model_path", type=str, default="memory_model.pt", help="已有模型路径")

    # 输入路径，可为单张图片或目录，具体取决于 mode。
    parser.add_argument("--input", type=str, help="输入图片路径或输入目录")

    # 输出目录，用于保存热力图、叠加图、结果 JSON 等。
    parser.add_argument("--output", type=str, help="输出目录，用于保存检测结果")

    # 正常 / 异常判定阈值。
    parser.add_argument("--threshold", type=float, default=1.0, help="异常判定阈值")

    # 阈值校准时使用的分位数。
    parser.add_argument("--quantile", type=float, default=0.99, help="阈值校准时使用的分位数")

    # 单图检测前是否手工框选 ROI。
    parser.add_argument("--select_roi", action="store_true", help="单图检测前手工选择 ROI 区域")

    # 热力图显示范围中，基于均值加标准差时的标准差倍数。
    parser.add_argument("--heatmap_std_scale", type=float, default=3.0, help="热力图显示范围中使用的标准差倍数")

    # 热力图显示范围上界的高分位数。
    parser.add_argument("--heatmap_quantile", type=float, default=0.999, help="热力图显示范围上界使用的分位数")

    # 阈值校准时最多采样多少个热力图数值，避免内存过大。
    parser.add_argument("--max_heatmap_samples", type=int, default=2_000_000, help="阈值校准时热力图采样上限")

    # 若开启，则阈值校准时跳过热力图统计，只计算全局分数阈值。
    parser.add_argument("--fast_calibrate", action="store_true", help="快速阈值校准：跳过热力图统计，只计算推荐阈值")

    # 若开启，则输出热力图时把阈值以下的位置强制置零。
    parser.add_argument("--heatmap_zero_below_threshold", action="store_true", help="将阈值以下的热力图值置零")

    # 训练时可保留的原始 embedding 上限。
    parser.add_argument("--max_embeddings", type=int, default=1200000, help="训练时最多保留多少原始 embedding")

    # 训练时随机裁剪尺度范围。
    parser.add_argument("--train_crop_scale_range", type=float, nargs=2, default=[0.7, 1.3], metavar=("MIN", "MAX"), help="训练随机裁剪缩放范围，例如 --train_crop_scale_range 0.7 1.3")

    # 训练时裁剪尺寸会对齐到该倍数。
    parser.add_argument("--train_crop_round_multiple", type=int, default=8, help="训练裁剪尺寸对齐倍数")

    # 训练时允许的最小裁剪尺寸。
    parser.add_argument("--train_min_crop_size", type=int, default=240, help="训练时最小裁剪尺寸")

    # 随机采样相关流程所用随机种子。
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")

    # 是否把 embedding 先流式写入磁盘而不是全部放内存。
    # 注意：这里使用的是 store_false 语义，传入该参数后反而会关闭磁盘流式写入。
    parser.add_argument("--stream_to_disk", action="store_false", help="是否将 embedding 流式写入磁盘；注意传入该参数会关闭此功能")

    # 流式写盘时的临时缓存目录。
    parser.add_argument("--stream_dir", type=str, default="./embedding_cache", help="embedding 流式缓存目录")

    # 在线训练时允许在候选池中保留的最大 embedding 数；大于 0 时启用在线压缩，避免缓存目录无限增长。
    parser.add_argument("--stream_max_embeddings", type=int, default=0, help="流式训练时允许暂存的最大 embedding 数，0 表示关闭在线压缩")

    # 在线压缩触发后保留的比例。
    parser.add_argument("--online_compress_ratio", type=float, default=0.5, help="在线压缩触发后保留比例，范围 (0, 1]")

    # 新 embedding 到当前候选池最近邻距离低于该阈值时直接丢弃。
    parser.add_argument("--online_novelty_threshold", type=float, default=0.0, help="在线去重阈值；大于 0 时仅保留与当前候选池足够不同的 embedding")

    # 训练前是否清空流式缓存目录。
    # 注意：这里使用的是 store_false 语义，传入该参数后反而会关闭清理。
    parser.add_argument("--cleanup_stream_dir", action="store_false", help="训练前是否清理缓存目录；注意传入该参数会关闭清理")

    # 是否开启训练时数据增强。
    # 注意：这里使用的是 store_false 语义，传入该参数后反而会关闭增强。
    parser.add_argument("--enable_train_augment", action="store_false", help="是否开启训练增强；注意传入该参数会关闭增强")

    # 原图保留次数。
    parser.add_argument("--aug_keep_original_count", type=int, default=1, help="每个裁剪块保留原图的次数")
    # 垂直翻转增强次数。
    parser.add_argument("--aug_vflip_count", type=int, default=0, help="每个裁剪块生成多少个垂直翻转样本")
    # 旋转增强次数。
    parser.add_argument("--aug_rotate_count", type=int, default=0, help="每个裁剪块生成多少个随机旋转样本")
    # 亮度增强次数。
    parser.add_argument("--aug_brightness_count", type=int, default=0, help="每个裁剪块生成多少个亮度扰动样本")
    # 对比度增强次数。
    parser.add_argument("--aug_contrast_count", type=int, default=0, help="每个裁剪块生成多少个对比度扰动样本")
    # 饱和度增强次数。
    parser.add_argument("--aug_saturation_count", type=int, default=0, help="每个裁剪块生成多少个饱和度扰动样本")
    # 模糊增强次数。
    parser.add_argument("--aug_blur_count", type=int, default=0, help="每个裁剪块生成多少个高斯模糊样本")
    # 噪声增强次数。
    parser.add_argument("--aug_noise_count", type=int, default=0, help="每个裁剪块生成多少个高斯噪声样本")
    # 颜色偏移增强次数。
    parser.add_argument("--aug_color_shift_count", type=int, default=0, help="每个裁剪块生成多少个颜色偏移样本")
    # Gamma 增强次数。
    parser.add_argument("--aug_gamma_count", type=int, default=0, help="每个裁剪块生成多少个 gamma 扰动样本")
    # 通道交换增强次数。
    parser.add_argument("--aug_channel_swap_count", type=int, default=0, help="每个裁剪块生成多少个通道交换样本")
    # 透视变换增强次数。
    parser.add_argument("--aug_perspective_count", type=int, default=0, help="每个裁剪块生成多少个透视变换样本")

    # 旋转增强角度范围。
    parser.add_argument("--aug_rotate_range", type=float, nargs=2, default=[-10.0, 10.0], metavar=("MIN", "MAX"), help="旋转增强角度范围，单位为度")
    # 亮度扰动幅度。
    parser.add_argument("--aug_brightness_limit", type=float, default=0.08, help="亮度增强的最大扰动幅度")
    # 对比度扰动幅度。
    parser.add_argument("--aug_contrast_limit", type=float, default=0.08, help="对比度增强的最大扰动幅度")
    # 饱和度扰动幅度。
    parser.add_argument("--aug_saturation_limit", type=float, default=0.08, help="饱和度增强的最大扰动幅度")
    # 模糊增强最小 sigma。
    parser.add_argument("--aug_blur_sigma_min", type=float, default=0.1, help="高斯模糊最小 sigma")
    # 模糊增强最大 sigma。
    parser.add_argument("--aug_blur_sigma_max", type=float, default=1.3, help="高斯模糊最大 sigma")
    # 噪声增强最小 sigma。
    parser.add_argument("--aug_noise_sigma_min", type=float, default=2.0, help="高斯噪声最小 sigma")
    # 噪声增强最大 sigma。
    parser.add_argument("--aug_noise_sigma_max", type=float, default=8.0, help="高斯噪声最大 sigma")
    # 颜色偏移上限。
    parser.add_argument("--aug_color_shift_limit", type=int, default=8, help="颜色偏移增强的最大通道偏移值")
    # Gamma 范围。
    parser.add_argument("--aug_gamma_range", type=float, nargs=2, default=[0.95, 1.05], metavar=("MIN", "MAX"), help="gamma 增强范围")
    # 透视变换强度。
    parser.add_argument("--aug_perspective_distortion", type=float, default=0.04, help="透视变换增强强度")

    # 追加正样本前是否先手工选择 ROI。
    parser.add_argument("--append_select_roi", action="store_true", help="追加正样本前手工选择 ROI")

    # 追加正样本时最多保留多少个 embedding，0 表示全保留。
    parser.add_argument("--append_max_embeddings", type=int, default=0, help="追加正样本时最多保留多少个 embedding，0 表示全部保留")

    # 追加正样本后是否再次压缩记忆库。
    parser.add_argument("--append_recompress_ratio", type=float, default=1.0, help="追加正样本后重新压缩的比例，1.0 表示不压缩")
    return parser


def build_engine(args):
    return VisionMemoryEngine(
        device=args.device,
        backbone=args.backbone,
        input_size=parse_tuple2(args.input_size, "input_size"),
        memory_ratio=args.memory_ratio,
        target_embed_dimension=args.target_embed_dimension,
        local_kernel=args.local_kernel,
        knn_neighbors=args.knn_neighbors,
        knn_backend=args.knn_backend,
        knn_query_chunk_size=args.knn_query_chunk_size,
        bm_bmodel_path=args.bm_bmodel_path,
        bm_device_id=args.bm_device_id,
        bm_db_chunk_size=args.bm_db_chunk_size,
        bm_graph_name=args.bm_graph_name,
        bm_query_input_name=args.bm_query_input_name,
        bm_database_input_name=args.bm_database_input_name,
        bm_output_name=args.bm_output_name,
        use_amp=args.use_amp,
        enable_train_augment=args.enable_train_augment,
        aug_keep_original_count=args.aug_keep_original_count,
        aug_vflip_count=args.aug_vflip_count,
        aug_rotate_count=args.aug_rotate_count,
        aug_brightness_count=args.aug_brightness_count,
        aug_contrast_count=args.aug_contrast_count,
        aug_saturation_count=args.aug_saturation_count,
        aug_blur_count=args.aug_blur_count,
        aug_noise_count=args.aug_noise_count,
        aug_color_shift_count=args.aug_color_shift_count,
        aug_gamma_count=args.aug_gamma_count,
        aug_channel_swap_count=args.aug_channel_swap_count,
        aug_perspective_count=args.aug_perspective_count,
        aug_rotate_range=parse_float_tuple2(args.aug_rotate_range, "aug_rotate_range"),
        aug_brightness_limit=args.aug_brightness_limit,
        aug_contrast_limit=args.aug_contrast_limit,
        aug_saturation_limit=args.aug_saturation_limit,
        aug_blur_sigma_min=args.aug_blur_sigma_min,
        aug_blur_sigma_max=args.aug_blur_sigma_max,
        aug_noise_sigma_min=args.aug_noise_sigma_min,
        aug_noise_sigma_max=args.aug_noise_sigma_max,
        aug_color_shift_limit=args.aug_color_shift_limit,
        aug_gamma_range=parse_float_tuple2(args.aug_gamma_range, "aug_gamma_range"),
        aug_perspective_distortion=args.aug_perspective_distortion,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.knn_backend == "bm" and not args.bm_bmodel_path:
        parser.error("knn_backend=bm requires --bm_bmodel_path")

    crop_size = parse_tuple2(args.crop_size, "crop_size")
    stride = parse_tuple2(args.stride, "stride") if args.stride is not None else None
    train_crop_scale_range = parse_float_tuple2(args.train_crop_scale_range, "train_crop_scale_range")
    engine = build_engine(args)

    if args.mode == "train":
        if not args.train_dir:
            parser.error("train mode requires --train_dir")
        size = engine.build_memory_bank(
            image_dir=args.train_dir,
            crop_size=crop_size,
            stride=stride,
            batch_size=args.batch_size,
            max_embeddings=args.max_embeddings,
            train_crop_scale_range=train_crop_scale_range,
            train_crop_round_multiple=args.train_crop_round_multiple,
            train_min_crop_size=args.train_min_crop_size,
            random_seed=args.random_seed,
            stream_to_disk=args.stream_to_disk,
            stream_dir=args.stream_dir,
            stream_max_embeddings=args.stream_max_embeddings,
            online_compress_ratio=args.online_compress_ratio,
            online_novelty_threshold=args.online_novelty_threshold,
            cleanup_stream_dir=args.cleanup_stream_dir,
            infer_long_side=args.infer_long_side,
        )
        print(f"Memory bank size: {size}")
        engine.save(args.save_model)
        return

    if args.mode == "calibrate_threshold":
        if not args.input:
            parser.error("calibrate_threshold mode requires --input")
        engine.load(args.model_path)
        thr = engine.calibrate_threshold(
            image_dir=args.input,
            crop_size=crop_size,
            stride=stride,
            quantile=args.quantile,
            heatmap_std_scale=args.heatmap_std_scale,
            heatmap_quantile=args.heatmap_quantile,
            max_heatmap_samples=args.max_heatmap_samples,
            detect_batch_size=args.detect_batch_size,
            infer_long_side=args.infer_long_side,
            fast_calibrate=args.fast_calibrate,
        )
        print(f"Recommended threshold = {thr:.6f}")
        engine.save(args.model_path)
        return

    if args.mode == "detect":
        if not args.input:
            parser.error("detect mode requires --input")
        ensure_dir(args.output)
        image_bgr = cv2.imread(args.input)
        if image_bgr is None:
            raise ValueError(f"Cannot read image: {args.input}")

        roi_xyxy = None
        if args.select_roi:
            roi_xyxy = select_roi_with_tk(image_bgr, window_title="SelectROI")
            if roi_xyxy is None:
                print("[INFO] 未选择 ROI，已取消检测")
                return

        engine.load(args.model_path)
        heatmap_path = os.path.join(args.output, f"{Path(args.input).stem}_overlay.jpg") if args.output else None
        start = time.time()

        if roi_xyxy is not None:
            x1, y1, x2, y2 = roi_xyxy
            roi_rgb = cv2.cvtColor(image_bgr[y1:y2, x1:x2].copy(), cv2.COLOR_BGR2RGB)
            is_anomaly, score, _ = engine.detect_image(
                image_rgb=roi_rgb,
                crop_size=crop_size,
                stride=stride,
                threshold=args.threshold,
                save_heatmap=heatmap_path,
                original_image_for_overlay=image_bgr,
                roi_xyxy=roi_xyxy,
                detect_batch_size=args.detect_batch_size,
                infer_long_side=args.infer_long_side,
                heatmap_zero_below_threshold=args.heatmap_zero_below_threshold,
            )
            end = time.time()
            print(f"Time: {end - start:.6f}")
            print(f"Image: {args.input}")
            print(f"ROI: {roi_xyxy}")
            print(f"Score: {score:.6f}")
            print(f"Threshold: {args.threshold:.6f}")
            print(f"Conclusion: {'ANOMALY' if is_anomaly else 'NORMAL'}")
            return

        is_anomaly, score, _ = engine.detect(
            image_path=args.input,
            crop_size=crop_size,
            stride=stride,
            threshold=args.threshold,
            save_heatmap=heatmap_path,
            detect_batch_size=args.detect_batch_size,
            infer_long_side=args.infer_long_side,
            heatmap_zero_below_threshold=args.heatmap_zero_below_threshold,
        )
        end = time.time()
        print(f"Time: {end - start:.6f}")
        print(f"Image: {args.input}")
        print(f"Score: {score:.6f}")
        print(f"Threshold: {args.threshold:.6f}")
        print(f"Conclusion: {'ANOMALY' if is_anomaly else 'NORMAL'}")
        return

    if args.mode == "detect_batch":
        if not args.input:
            parser.error("detect_batch mode requires --input")
        engine.load(args.model_path)
        results = engine.detect_batch(
            image_dir=args.input,
            crop_size=crop_size,
            stride=stride,
            threshold=args.threshold,
            output_dir=args.output,
            detect_batch_size=args.detect_batch_size,
            infer_long_side=args.infer_long_side,
            heatmap_zero_below_threshold=args.heatmap_zero_below_threshold,
        )
        total = len(results)
        abnormal = sum(1 for v in results.values() if v.get("is_anomaly") is True)
        failed = sum(1 for v in results.values() if v.get("is_anomaly") is None)
        print(f"Total: {total}")
        print(f"Abnormal: {abnormal}")
        print(f"Failed: {failed}")
        return

    if args.mode == "append_positive":
        if not args.input:
            parser.error("append_positive mode requires --input")
        engine.load(args.model_path)
        input_path = Path(args.input)
        image_paths = list_images(str(input_path)) if input_path.is_dir() else [str(input_path)] if input_path.is_file() else []
        if not image_paths:
            raise ValueError(f"No images found in {args.input}")

        all_new_embeddings = []
        for img_path in tqdm(image_paths, desc="Appending positive samples"):
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                print(f"[WARN] Cannot read image: {img_path}")
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            if args.append_select_roi:
                roi_xyxy = select_roi_with_tk(image_bgr, window_title="AppendPositiveROI")
                if roi_xyxy is None:
                    print(f"[INFO] Skip {img_path}, ROI not selected.")
                    continue
                x1, y1, x2, y2 = roi_xyxy
                image_rgb = image_rgb[y1:y2, x1:x2].copy()
            try:
                all_new_embeddings.append(
                    engine.extract_image_embeddings(
                        image_rgb=image_rgb,
                        crop_size=crop_size,
                        stride=stride,
                        detect_batch_size=args.detect_batch_size,
                        infer_long_side=args.infer_long_side,
                    )
                )
            except Exception as exc:
                print(f"[WARN] Failed processing {img_path}: {exc}")

        if not all_new_embeddings:
            raise ValueError("No new positive embeddings extracted.")

        final_n = engine.append_positive_embeddings(
            embeddings=torch.cat(all_new_embeddings, dim=0).float(),
            max_append_embeddings=args.append_max_embeddings,
            random_seed=args.random_seed,
            recompress_ratio=args.append_recompress_ratio,
        )
        engine.save(args.model_path)
        print(f"Append positive finished. New memory bank size: {final_n}")


def read_img(filename: str | os.PathLike[str]):
    global _original_imread
    height = 640
    img = _original_imread(filename)
    h, w = img.shape[:2]
    if h != height:
        width = int(w * height / h)
        img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
    return img

_original_imread = cv2.imread


if __name__ == "__main__":
    cv2.imread = read_img
    main()
