import os
import csv
import onnxruntime as ort
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

###############################################
# 0. Required functions, including image preprocessing, EER calculation, etc.
###############################################

# Equal Error Rate calculation function
def calc_eer(distances, label, threshold_list=None):
    """
    Calculate the Equal Error Rate (EER), as well as additional info such as TAR when FAR=0.01.
    :param distances: list of cosine similarities, [batch_size]
    :param label: list of labels, [batch_size]; 1 means same class, 0 means different class
    :return: intra_cnt_final, inter_cnt_final, intra_len_final, inter_len_final, eer, bestThresh, minV, tar_far001
    """
    distances_np = np.array([d.item() if isinstance(d, torch.Tensor) else d for d in distances])
    label_np = np.array(label)

    batch_size = label_np.shape[0]
    max_dist = np.max(distances_np)
    min_dist = np.min(distances_np)

    if threshold_list is None:
        threshold_list = np.linspace(min_dist, max_dist, num=100)

    minV = 1e10
    minV_far001 = 1e10
    bestThresh = 0
    tar_2f = 0.0

    intra_cnt_final = 0
    inter_cnt_final = 0
    intra_len_final = 0
    inter_len_final = 0
    eer = 0.0

    for threshold in threshold_list:
        intra_cnt = 0
        intra_len = 0
        inter_cnt = 0
        inter_len = 0
        # Traverse all distances to calculate FRR and FAR
        for i in range(batch_size):
            if label_np[i] == 1:
                intra_len += 1
                if distances_np[i] < threshold:
                    intra_cnt += 1
            else:
                inter_len += 1
                if distances_np[i] > threshold:
                    inter_cnt += 1

        # FRR / FAR
        fr = intra_cnt / (intra_len if intra_len > 0 else 1)
        fa = inter_cnt / (inter_len if inter_len > 0 else 1)

        # If the difference between FRR and FAR is smaller, update the EER
        if abs(fr - fa) < minV:
            minV = abs(fr - fa)
            eer = (fr + fa) / 2
            bestThresh = threshold

            intra_cnt_final = intra_cnt
            inter_cnt_final = inter_cnt
            intra_len_final = intra_len
            inter_len_final = inter_len

        # Also record "TAR = 1 - FRR" when FAR=0.01
        if abs(0.01 - fa) < minV_far001:
            minV_far001 = abs(0.01 - fa)
            tar_2f = 1 - fr

    return intra_cnt_final, inter_cnt_final, intra_len_final, inter_len_final, eer, bestThresh, minV, tar_2f


transform_notrans = T.Compose([
    T.Resize((320, 320)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_image(img_path):
    """
    Open the image and perform preprocessing consistent with training/testing, returning a numpy array (1,3,320,320).
    """
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = transform_notrans(img)
    # Add batch dimension
    img = img.unsqueeze(0)  # [1, 3, 320, 320]
    return img.numpy()

###############################################
# 1. Define a simple test dataset class (for demonstration only)
###############################################
class TestDataset(object):
    """
    For a CSV file with lines such as:
       68_1.jpg,68_2.jpg,1
       68_1.jpg,73_2.jpg,0
       ...
    and two modalities (example: fingerprint/palmprint),
    construct `prefixs` (unique image names) and `query` (pairs).
    If you have only one modality, you can remove the unused code.
    """
    def __init__(self, csv_file, finger_root, palm_root):
        """
        :param csv_file: The test set file containing all (imgA, imgB, label)
        :param finger_root: The root directory for finger images (or finger vein, etc.)
        :param palm_root:   The root directory for palm images
        """
        self.finger_root = finger_root
        self.palm_root = palm_root
        # Read CSV
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = list(reader)

        # `prefixs` collects all appeared image names (duplicates removed)
        all_imgs = set()
        self.query = []
        for row in lines:
            imgA, imgB, lab = row[0].strip(), row[1].strip(), int(row[2].strip())
            self.query.append((imgA, imgB, lab))
            all_imgs.add(imgA)
            all_imgs.add(imgB)

        self.prefixs = list(all_imgs)

    def __len__(self):
        return len(self.prefixs)

    def __getitem__(self, idx):
        """
        Return the two modalities (finger_modality, palm_modality) corresponding to one prefix,
        as well as the image filename.
        Both finger_modality/palm_modality are numpy arrays with shape (1,3,320,320).
        """
        filename = self.prefixs[idx]
        finger_path = os.path.join(self.finger_root, filename)
        palm_path   = os.path.join(self.palm_root,   filename)

        # Load each
        x_finger = load_image(finger_path)  # (1,3,320,320)
        x_palm   = load_image(palm_path)    # (1,3,320,320)
        # Return the two modalities plus the filename
        return (x_finger, x_palm), filename


###############################################
# 2. Main inference process: load ONNX, iterate through dataset, and compute EER
###############################################
def main():
    # 2.1 Specify the ONNX model path and test dataset information
    onnx_model_path = "export_SCUT_NFPH_v1.onnx"
    test_csv_path   = "SCUT_NFPH_v1_pair_test.csv"          # Change to your actual test set path
    finger_root     = "Normalized_Finger/SCUT_NFPH_v1/Concatenated_Aligned_finger_(AF)/"   # Change to your actual finger image directory
    palm_root       = "Normalized_Palm/SCUT_NFPH_v1/"     # Change to your actual palm image directory

    # 2.2 Create an onnxruntime Session
    session = ort.InferenceSession(onnx_model_path)

    # Optional: view model input information (for debugging)
    print("=== Model Inputs ===")
    for idx, inp in enumerate(session.get_inputs()):
        print(f"Input {idx} - name: {inp.name}, shape: {inp.shape}, type: {inp.type}")

    # 2.3 Build the test dataset and create a dictionary <filename -> feature vector>
    dataset = TestDataset(csv_file=test_csv_path,
                          finger_root=finger_root,
                          palm_root=palm_root)

    features_dict = {}
    print("\n[INFO] Generating embeddings for each unique file ...")
    for i in range(len(dataset)):
        # Read data
        (x_finger, x_palm), filename = dataset[i]

        # Build feed_dict
        # Make sure the order and names match the model inputs
        feed_dict = {
            session.get_inputs()[0].name: x_finger,  # finger
            session.get_inputs()[1].name: x_palm     # palm
        }

        # Run inference
        outputs = session.run(None, feed_dict)
        emb = outputs[6]  # numpy array of shape: (1, feat_dim)

        # Convert to torch tensor for subsequent cosine_similarity
        emb_tensor = torch.from_numpy(emb[0])  # (feat_dim,)
        features_dict[filename] = emb_tensor

    # 2.4 Traverse the CSV queries, use the extracted features to compute cosine similarity
    distances = []
    labels = []
    print("\n[INFO] Computing pairwise distances from query ...")
    for (imgA, imgB, lab) in dataset.query:
        # Retrieve features from the dictionary
        featA = features_dict[imgA]  # (feat_dim,)
        featB = features_dict[imgB]  # (feat_dim,)

        # Calculate similarity (using cosine; the larger the value, the more similar)
        cos_sim = F.cosine_similarity(featA.unsqueeze(0), featB.unsqueeze(0)).item()

        distances.append(cos_sim)
        labels.append(lab)

    # 2.5 Call `calc_eer`
    (intra_cnt, inter_cnt, intra_len, inter_len,
     eer, bestThresh, minV, tar_2f) = calc_eer(distances, labels)

    # 2.6 Print the results
    print("========================================")
    print("[RESULT] EER: {:.4f}, bestThresh: {:.4f}, |FR-FA|: {:.6f}, TAR@FAR=0.01: {:.4f}".format(
        eer, bestThresh, minV, tar_2f
    ))
    print("[RESULT] Intra: {}/{}  Inter: {}/{}".format(
        intra_cnt, intra_len, inter_cnt, inter_len
    ))

if __name__ == "__main__":
    main()
