import glob
import json
import os

import clip
import numpy as np
import streamlit as st
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, BatchSampler
from tqdm.notebook import tqdm

DEBUG = True

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
EPOCH = 2


class Training():
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)

    def my_func(self, files, img_paths, img_text_pairs):
        print("hello there")
        print(files)

        ################
        # do something #
        ################

        img_paths, img_text_pairs = self.make_temp_dataset()

        train_img_paths, test_img_paths = train_test_split(img_paths, test_size=0.2, random_state=42)
        d_train = {k: img_text_pairs[k] for k in train_img_paths}
        d_test = {k: img_text_pairs[k] for k in test_img_paths}
        train_dataset = MemeDataset(d_train, self.preprocess)
        test_dataset = MemeDataset(d_test, self.preprocess)

        train_labels = torch.tensor([item[2] for item in train_dataset])
        train_sampler = BalancedBatchSampler(train_labels, BATCH_SIZE, 1)
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)

        test_labels = torch.tensor([item[2] for item in test_dataset])
        test_sampler = BalancedBatchSampler(test_labels, BATCH_SIZE, 1)
        test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler)

        for i, item in enumerate(train_sampler):
            labels = []
            for idx in item:
                label = train_dataset[idx][2]
                labels.append(label)
            break
        if device == "cpu":
            self.model.float()

        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader) * EPOCH)

        best_te_loss = 1e5
        best_ep = -1
        for epoch in range(EPOCH):
            st.spinner('Please wait while training...')
            print(f"running epoch {epoch}, best test loss {best_te_loss} after epoch {best_ep}")
            step = 0
            tr_loss = 0
            self.model.train()
            pbar = tqdm(train_dataloader, leave=False)
            for batch in pbar:
                step += 1
                optimizer.zero_grad()

                images, texts, _ = batch
                images = images.to(device)
                texts = clip.tokenize(texts).to(device)
                #         print(images.shape, texts.shape)
                logits_per_image, logits_per_text = self.model(images, texts)
                ground_truth = torch.arange(BATCH_SIZE).to(device)

                total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                total_loss.backward()
                tr_loss += total_loss.item()
                if device == "cpu":
                    optimizer.step()
                    scheduler.step()
                else:
                    self.convert_models_to_fp32(self.model)
                    optimizer.step()
                    scheduler.step()
                    clip.model.convert_weights(self.model)
                pbar.set_description(f"train batchCE: {total_loss.item()}", refresh=True)
            tr_loss /= step

            step = 0
            te_loss = 0

            with torch.no_grad():
                self.model.eval()
                test_pbar = tqdm(test_dataloader, leave=False)
                for batch in test_pbar:
                    step += 1
                    images, texts, _ = batch
                    images = images.to(device)
                    texts = clip.tokenize(texts).to(device)
                    logits_per_image, logits_per_text = self.model(images, texts)
                    ground_truth = torch.arange(BATCH_SIZE).to(device)

                    total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text,
                                                                                      ground_truth)) / 2
                    te_loss += total_loss.item()
                    test_pbar.set_description(f"test batchCE: {total_loss.item()}", refresh=True)
                te_loss /= step

            if te_loss < best_te_loss:
                best_te_loss = te_loss
                best_ep = epoch
                torch.save(self.model.state_dict(), "./train_sample/best_model_test.pt")
            print(f"epoch {epoch}, tr_loss {tr_loss}, te_loss {te_loss}")
        torch.save(self.model.state_dict(), "./train_sample/last_model_test.pt")

        st.sucess('Done.')

    def convert_models_to_fp32(model):
        for p in model.parameters():
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()

    def make_temp_dataset(self):
        IMG_ROOT = "C:/GitHub_clone/LIAM/custom_data/images/pyh/"
        JSON_ROOT = "C:/GitHub_clone/LIAM/custom_data/images/pyh_json/"
        img_paths = glob.glob(os.path.join(IMG_ROOT, "*.jpg"))

        d = {}
        for i, img_path in enumerate(img_paths):
            name = img_path.replace('\\', '/').split("/")[-1].split(".")[0]
            with open(os.path.join(JSON_ROOT, name + ".json"), "r") as f:
                captions = json.load(f)
                temp = []
                for cap in captions:
                    if "http" not in (cap[0] + ' ' + cap[1]) and len(cap[0] + ' ' + cap[1]) >= 8 and len(
                            cap[0] + ' ' + cap[1]) <= 70:
                        temp.append(cap[0] + ' ' + cap[1])
                d[img_path] = temp
        return img_paths, d


class MemeDataset(Dataset):
    def __init__(self, data, preprocess):
        self.preprocess = preprocess
        self.img_paths = []
        self.captions = []
        for img_path, captions in data.items():
            for cap in captions:
                self.img_paths.append(img_path)
                self.captions.append(cap)
        self.processed_cache = {}
        for img_path in data:
            self.processed_cache[img_path] = self.preprocess(Image.open(img_path))
        self.img_paths_set = list(data.keys())
        self.path2label = {path: self.img_paths_set.index(path) for path in self.img_paths_set}

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = self.processed_cache[img_path]
        caption = self.captions[idx]
        label = self.path2label[img_path]
        return image, caption, label


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


def main() -> None:
    st.image('./imgs/logo.png', width=300)
    st.title("학습을 위한 이미지 파일을 업로드 해주세요.")
    selfTraining = Training()
    runbtn = st.button('Train')

    uploaded_files = st.file_uploader("Choose image files", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

    text_pairs = []
    img_text_pairs = {}
    img_paths = []
    img_root = 'C:/GitHub_clone/LIAM/custom_data/images/pyh/'

    if uploaded_files is not None:
        for i, uploaded_file in enumerate(uploaded_files):
            # tfile = tempfile.NamedTemporaryFile(delete=False)
            # tfile.write(uploaded_file.read())
            # print(tfile.name)

            label = f"Description for image {str(i)}"
            st.image(uploaded_file, caption=uploaded_file.name, width=150)
            text_pair = st.text_input(label, '')
            text_pairs.append(text_pair)
            full_path = img_root + uploaded_file.name
            img_paths.append(full_path)
            img_text_pairs[full_path] = text_pair

        # Call Train API

        if "runbtn_state" not in st.session_state:
            st.session_state.runbtn_state = False

        if runbtn or st.session_state.runbtn_state:
            st.session_state.runbtn_state = True

            if DEBUG:
                st.write(uploaded_files)

            selfTraining.my_func(uploaded_files, img_paths, img_text_pairs)  # Your function


if __name__ == "__main__":
    main()
