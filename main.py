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
    parser = argparse.ArgumentParser(description="Memory-based visual anomaly tool")
    parser.add_argument("mode", choices=["train", "detect", "detect_batch", "calibrate_threshold", "append_positive"])

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--input_size", type=int, nargs=2, default=[240, 240])
    parser.add_argument("--crop_size", type=int, nargs=2, default=[160, 160])
    parser.add_argument("--stride", type=int, nargs=2, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--detect_batch_size", type=int, default=8)
    parser.add_argument("--local_kernel", type=int, default=3)
    parser.add_argument("--memory_ratio", type=float, default=0.002)
    parser.add_argument("--target_embed_dimension", type=int, default=1024)
    parser.add_argument("--knn_neighbors", type=int, default=1)
    parser.add_argument("--knn_backend", type=str, default="auto", choices=["auto", "torch", "sklearn", "faiss"])
    parser.add_argument("--knn_query_chunk_size", type=int, default=8192)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--infer_long_side", type=int, default=0)

    parser.add_argument("--train_dir", type=str)
    parser.add_argument("--save_model", type=str, default="memory_model.pt")
    parser.add_argument("--model_path", type=str, default="memory_model.pt")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--select_roi", action="store_true")
    parser.add_argument("--heatmap_std_scale", type=float, default=3.0)
    parser.add_argument("--heatmap_quantile", type=float, default=0.999)
    parser.add_argument("--max_heatmap_samples", type=int, default=2_000_000)
    parser.add_argument("--heatmap_zero_below_threshold", action="store_true")

    parser.add_argument("--max_embeddings", type=int, default=1200000)
    parser.add_argument("--train_crop_scale_range", type=float, nargs=2, default=[0.7, 1.3])
    parser.add_argument("--train_crop_round_multiple", type=int, default=8)
    parser.add_argument("--train_min_crop_size", type=int, default=240)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--stream_to_disk", action="store_false")
    parser.add_argument("--stream_dir", type=str, default="./embedding_cache")
    parser.add_argument("--cleanup_stream_dir", action="store_false")

    parser.add_argument("--enable_train_augment", action="store_false")
    parser.add_argument("--aug_keep_original_count", type=int, default=1)
    parser.add_argument("--aug_vflip_count", type=int, default=0)
    parser.add_argument("--aug_rotate_count", type=int, default=0)
    parser.add_argument("--aug_brightness_count", type=int, default=0)
    parser.add_argument("--aug_contrast_count", type=int, default=0)
    parser.add_argument("--aug_saturation_count", type=int, default=0)
    parser.add_argument("--aug_blur_count", type=int, default=0)
    parser.add_argument("--aug_noise_count", type=int, default=0)
    parser.add_argument("--aug_color_shift_count", type=int, default=0)
    parser.add_argument("--aug_gamma_count", type=int, default=0)
    parser.add_argument("--aug_channel_swap_count", type=int, default=0)
    parser.add_argument("--aug_perspective_count", type=int, default=0)

    parser.add_argument("--aug_rotate_range", type=float, nargs=2, default=[-10.0, 10.0])
    parser.add_argument("--aug_brightness_limit", type=float, default=0.08)
    parser.add_argument("--aug_contrast_limit", type=float, default=0.08)
    parser.add_argument("--aug_saturation_limit", type=float, default=0.08)
    parser.add_argument("--aug_blur_sigma_min", type=float, default=0.1)
    parser.add_argument("--aug_blur_sigma_max", type=float, default=1.3)
    parser.add_argument("--aug_noise_sigma_min", type=float, default=2.0)
    parser.add_argument("--aug_noise_sigma_max", type=float, default=8.0)
    parser.add_argument("--aug_color_shift_limit", type=int, default=8)
    parser.add_argument("--aug_gamma_range", type=float, nargs=2, default=[0.95, 1.05])
    parser.add_argument("--aug_perspective_distortion", type=float, default=0.04)

    parser.add_argument("--append_select_roi", action="store_true")
    parser.add_argument("--append_max_embeddings", type=int, default=0)
    parser.add_argument("--append_recompress_ratio", type=float, default=1.0)
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
