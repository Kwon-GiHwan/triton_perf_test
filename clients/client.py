import numpy as np
import tritonclient.http as httpclient


def main():
    url = "localhost:8000"
    model_name = "simple_model"

    client = httpclient.InferenceServerClient(url=url)

    # 서버 상태 확인
    if not client.is_server_live():
        print("Server is not live")
        return

    print(f"Model '{model_name}' ready: {client.is_model_ready(model_name)}")

    # 입력 데이터 생성
    input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

    inputs = [
        httpclient.InferInput("INPUT0", input_data.shape, "FP32"),
    ]
    inputs[0].set_data_from_numpy(input_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    # 추론 요청
    result = client.infer(model_name, inputs, outputs=outputs)
    output_data = result.as_numpy("OUTPUT0")

    print(f"Input:  {input_data}")
    print(f"Output: {output_data}")


if __name__ == "__main__":
    main()
