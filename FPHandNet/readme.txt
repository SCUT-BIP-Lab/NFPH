FPHandNet

Pretrained Weights

The weights of FPHandNet are provided in ONNX format to facilitate inference.

Inference Code Example：

import numpy as np
import onnxruntime as ort
import torch

# 1. Load the ONNX model and check its inputs
onnx_model_path = "export_SCUT_NFPH_v1.onnx"
session = ort.InferenceSession(onnx_model_path)

inputs_info = session.get_inputs()
for idx, inp in enumerate(inputs_info):
    print(f"Input {idx} - name: {inp.name}, shape: {inp.shape}, type: {inp.type}")

# 2. Prepare sample input data as PyTorch tensors
x1_tensor = torch.randn(1, 3, 320, 320)
x2_tensor = torch.randn(1, 3, 320, 320)

# Convert tensors to NumPy arrays
x1_np = x1_tensor.numpy()
x2_np = x2_tensor.numpy()

# 3. Create the feed dictionary for the session
feed_dict = {
    session.get_inputs()[0].name: x1_np,
    session.get_inputs()[1].name: x2_np
}

# 4. Run inference
outputs = session.run(None, feed_dict)
# Passing 'None' fetches data from all output nodes.

 5. Check the outputs
print("=================================")
print("[INFO] Got ONNX model outputs:")
for i, out in enumerate(outputs):
    print(f"  Output {i} shape = {out.shape}, dtype = {out.dtype}")

Example Console Output：
Input 0 - name: input.1, shape: [1, 3, 320, 320], type: tensor(float)
Input 1 - name: input.121, shape: [1, 3, 320, 320], type: tensor(float)
=================================
[INFO] Got ONNX model outputs:
  Output 0 shape = (1, 1280), dtype = float32
  Output 1 shape = (1, 1280), dtype = float32
  Output 2 shape = (1, 1280), dtype = float32
  Output 3 shape = (1, 1280), dtype = float32
  Output 4 shape = (1, 1280), dtype = float32
  Output 5 shape = (1, 1280), dtype = float32
  Output 6 shape = (1, 7680), dtype = float32

Explanation：
Outputs 0–4 represent feature vectors corresponding to different finger regions.

Output 5 is the feature vector for the palm region.

Output 6 is an overall (integrated) feature vector for the entire hand.

All these features can be combined or used individually based on the specific requirements of your application.

If you encounter any issues while using the model weights we provided, please leave an issue.
The code for the FPHand network architecture will be made publicly available promptly upon acceptance.




