import time

def test_store_init():

    from store_core import TrainRoofAnomalyStore
    from store_core.schemas import RuntimeOptions

    store = TrainRoofAnomalyStore(
        root_dir="./store_data",
        autostart_service=True,
        service_port=55555,
        yolo_conf_threshold=0.8
    )

    print(store.service_info)
    return store


def test_store_train(store):
    from store_core.schemas import RuntimeOptions

    def on_progress(event: dict):
        print(event)

    options = RuntimeOptions(
        device="cuda",
        knn_backend="faiss",
        crop_size=(640, 640),
        stride=(512, 512),
        threshold_quantile=0.99,
    )

    result = store.train_model(
        model_name="木板车顶模型",
        image_dir="templates",
        runtime_options=options,
        calibrate_dir=None,
        progress_callback=on_progress,
    )

    print(result["model_id"])
    return result["model_id"]


def test_store_predict(store, model_id=None):
    result = store.detect_image(
        model_id="model_e8ad3cf30cd14a10" if model_id is None else model_id,
        image_path="test_imgs/1.jpg",
        include_heatmap_base64=True,
    )

    print(result)


if __name__ == '__main__':
    model_id = "model_e8ad3cf30cd14a10"
    store = test_store_init()
    # model_id = test_store_train(store)
    test_store_predict(store, model_id=model_id)

    while True:
        time.sleep(1)
