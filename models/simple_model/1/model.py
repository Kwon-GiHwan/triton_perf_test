import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            input_data = input_tensor.as_numpy()

            output_data = input_data * 2.0

            output_tensor = pb_utils.Tensor("OUTPUT0", output_data.astype(np.float32))
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        return responses

    def finalize(self):
        pass
