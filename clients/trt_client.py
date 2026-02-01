import numpy as np
import tritonclient.http as httpclient


def main():
    url = "localhost:8000"
    model_name = "my_trt_model"

    client = httpclient.InferenceServerClient(url=url)

    if not client.is_server_live():
        print("Server is not live")
        return

    print(f"Model '{model_name}' ready: {client.is_model_ready(model_name)}")

    # 입력 데이터 생성 (batch=1, 3x224x224 이미지)
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

    inputs = [
        httpclient.InferInput("input", input_data.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(input_data)

    outputs = [
        httpclient.InferRequestedOutput("output"),
    ]

    result = client.infer(model_name, inputs, outputs=outputs)
    output_data = result.as_numpy("output")

    print(f"Input shape:  {input_data.shape}")
    print(f"Output shape: {output_data.shape}")
    print(f"Top-5 indices: {np.argsort(output_data[0])[-5:][::-1]}")


if __name__ == "__main__":
    main()
