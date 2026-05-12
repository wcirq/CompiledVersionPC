from store_infer import list_models, run_inference


def test_list_inference_models():
    print(list_models())


def test_run_fire_smoke():
    result = run_inference(
        model_name="fire_smoke",
        image_path="test_imgs/2.jpg",
        conf_threshold=0.25,
        include_visualization_base64=True,
    )
    print(result)


if __name__ == "__main__":
    test_list_inference_models()
    test_run_fire_smoke()
