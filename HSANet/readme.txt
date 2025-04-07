HSANet (handsegmentation and alignment network) is a real-time part-level hand semantic segmentation network built upon the PaddleSeg framework. 

More experimental details and results are available in our paper:

> **"Normalized-Full-Palmar-Hand: Towards More Accurate Hand-Based Multimodal Biometrics."**

When using HSANet, please place the HSANet code in the PaddleSeg/paddleseg/models/ directory and import it in PaddleSeg/paddleseg/models/__init__.py using "from .HSANet import HSANet". At the same time, copy the contents of layers/tensor_fusion.py to the folder with the same name in the original framework, and add "from .tensor_fusion import SimpleAddFusion" to PaddleSeg/paddleseg/models/layers/__init__.py.