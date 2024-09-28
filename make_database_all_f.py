import torchaudio
import numpy as np
import faiss
import faiss.contrib.torch_utils
import soundfile as sf
from tqdm import tqdm
import pickle
import re
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.multiprocessing as mp
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

device = torch.device('cpu')
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').to(device)

def extract_feature(tsrs, path=None):
    # data, samplerate = sf.read(path)
    if os.path.exists(path.replace('.wav', '-sv.pt')):
        return torch.load(path.replace('.wav', '-sv.pt'), map_location=device)
    data = feature_extractor(tsrs, sampling_rate=16000,padding=False, return_tensors="pt").to(device)
    embeddings = model(**data).embeddings
    torch.save(embeddings, path.replace('.wav', '-sv.pt'))
    return embeddings
    # torch.cuda.empty_cache()
    # return embeddings



def kmeans_speaker(speaker, speaker_df):
    df = speaker_df[speaker_df['speaker'] == speaker]
    paths = df['path'].tolist()
    if len(paths) >= 150:
        select_path = random.sample(paths, k =150)
    else:
        select_path = paths
    
    # for topic in topics:
    #     tmp_files = os.listdir(root_path + '/' + topic)
    #     tmp_files = [root_path + '/' + topic + '/' + f for f in tmp_files if f.endswith('.pt')]
    #     pt_files += tmp_files
    # 将 tensor_list 视图转换为二维并复制到 CPU
    pt_files = [torch.load(path, map_location=torch.device('cpu')) for path in select_path]
    pt_features = []
    for pt in pt_files:
        tmp_pt = torch.mean(pt, dim = 1)
        pt_features.append(tmp_pt)
    pt_features = torch.cat(pt_features, dim=0)
    # pt_features = pt_features.squeeze()
    tmp = pt_features.detach()
    
    # 执行 kmeans 聚类
    kmeans_helper = faiss.Kmeans(768, k=1, seed=12345)
    kmeans_helper.train(tmp)
    cluster_centers = kmeans_helper.centroids
    return torch.tensor(cluster_centers)



random.seed(12345)

audio_record = []

def merge_speaker(speaker, speaker_df):
    # mp.set_start_method('spawn', force=True)
    df = speaker_df[speaker_df['speaker'] == speaker]
    paths = df['path'].tolist()
    # MyDataset = AudioDataset(paths)
    # loader = DataLoader(MyDataset, batch_size = 360, shuffle = True, num_workers = 8,)
    # for batch in loader:
    global audio_record
    if len(paths) >= 150:
        select_path = random.sample(paths, k = 150)
        audio_record += [p + '\n' for p in select_path]
    else:
        select_path = paths
    # tensors = [torchaudio.load(file.replace('.pt', '.wav'), map_location=torch.device('cuda:4')).detach() for file in select_path]
    # tensors = [torchaudio.load(file.replace('.pt', '.wav'))[0] for file in select_path]
    # for p in select_path:
    #     tensor = sf.read(p)[0]
    #     extract_feature(tensor, p)
    # tensors = [sf.read(p)[0] for p in select_path]
    # tens1 = tensors[:180]
    # tens2 = tensors[180:]
    # tens1 = extract_feature(tens1)
    # tens2 = extract_feature(tens2)
    # for p in select_path:
    #     extract_feature(sf.read(p)[0], p)
    tensors = [extract_feature(sf.read(p)[0], p)  for p in select_path]
    # tensors = []
    # for p in select_path:
    #     tmp = torch.load(p, map_location=torch.device('cuda:4'))
    #     tensors.append(tmp)
        # tensors = torch.cat(batch.view(-1, 768), dim=1)
    # return None
    tensors = torch.cat(tensors, dim = 0)
    tensors = torch.mean(tensors, dim = 0)
    return tensors
    
        # break
    # return tensors


def normalize_L2(query):
    # 计算每个向量的 L2 范数
    norms = torch.norm(query, p=2, dim=1, keepdim=True)
    # 将每个向量除以其 L2 范数
    normalized_query = query / norms
    return normalized_query


import gc
res = faiss.StandardGpuResources()  # use a single GPU
def make_dataset():
        # 设置聚类参数
    # res = faiss.StandardGpuResources()
    d = 1 * 512
    
    df = pd.read_csv('all_modified_4.csv', encoding='utf-8')
    speaker_df = pd.read_csv('/mnt/data1/liupeizhuo/detect/final_new/speaker_split_wav.csv')
    speaker_df = speaker_df.sample(frac=1, random_state=1234).reset_index(drop=True)
    df = df[df['misinformation'] == 1]
    # name_set = list()
    tensor_list = []
    name_list = []
    speaker_list = set(df['speaker'].tolist())
    speaker_list = list(speaker_list)
    print(len(speaker_list))
    for speaker in tqdm(speaker_list):
    # for i in tqdm(range(len(df))):
    #     speaker = df.iloc[i]['speaker']
        # if speaker in name_set:
        #     continue

        # tmp_path = df.iloc[i]['path'].split('/')[:-1]
        # tmp_path = '/'.join(tmp_path)
        # f = kmeans_speaker(speaker, speaker_df)
        # merge_speaker(speaker, speaker_df)
        f = merge_speaker(speaker, speaker_df)
        # torch.cuda.empty_cache()
        tensor_list.append(f)
        name_list.append(speaker)
    # exit()
    tensor_list = torch.cat([t.unsqueeze(0) for t in tensor_list], dim = 0)#.cpu()
    torch.save(tensor_list, f'/mnt/data1/liupeizhuo/detect/final_new/150_cosine_va_clean/index_data_uncompress.pt')
    # tensor_list = torch.load('/mnt/data1/liupeizhuo/detect/final_new/l2_database/index_data.pt')
    tensor_list = torch.nn.functional.normalize(tensor_list, dim=-1)
    tensor_list = normalize_L2(tensor_list)
    # faiss.normalize_L2(tensor_list.numpy())
    # tensor_list = torch.tensor(tensor_list)
    # index = faiss.IndexFlatL2(d)
    index = faiss.IndexFlatIP(d)
    index = faiss.index_cpu_to_gpu(res, 5, index)
    index.add(tensor_list)
    index = faiss.index_gpu_to_cpu(index)
    torch.save(tensor_list, f'/mnt/data1/liupeizhuo/detect/final_new/150_cosine_va_clean/index_data.pt')
    torch.save(speaker_list, f'/mnt/data1/liupeizhuo/detect/final_new/150_cosine_va_clean/index_name')
    faiss.write_index(index, f'/mnt/data1/liupeizhuo/detect/final_new/150_cosine_va_clean/names.index')
    print("Saved!")
    # return index

make_dataset()

with open('/mnt/data1/liupeizhuo/detect/final_new/150_cosine_va_clean/audio_record_150.txt', 'w') as fw:
    fw.writelines(audio_record)

    # ex_f = list(ex_f['path'])
    # ex_f = random.sample(ex_f, k=int(len(ex_f)/10))
    # cluster_record = list(set(ex_f))
    # cur_dataset = MyDataset(cluster_record)
        
    # batch = int(len(cluster_record)/30)
    # print(batch)
    # cluster_loader = DataLoader(cur_dataset, batch_size=batch, shuffle=True)
    # i = 0

    # for tensor_list in tqdm(cluster_loader):
    #     with torch.no_grad():
    #         tmp = None
    #         print(f"start kmeans {i + 1}")
            
    #         # 确保 tensor_list 是在 CPU 上的
    #         tensor_list = tensor_list.cpu()
            
    #         # 将 tensor_list 视图转换为二维并复制到 CPU
    #         tmp = tensor_list.view(batch, -1).cpu()
            
    #         # 执行 kmeans 聚类
    #         kmeans_helper = faiss.Kmeans(d, k=num_clusters, seed=1234)
    #         kmeans_helper.train(tmp.numpy())
    #         cluster_centers = kmeans_helper.centroids
            
    #         # 分配簇标签
    #         _, labels = kmeans_helper.assign(tmp.numpy())
            
    #         # 保存标签和聚类中心
    #         torch.save(labels, f'/mnt/data3/liupeizhuo/datasets/cluster_centers/cluster_labels_{i}')
    #         torch.save(cluster_centers, f'/mnt/data3/liupeizhuo/datasets/cluster_centers/cluster_centers_{i}')
    #         print('save well!')
            
    #         # 删除张量
    #         del tmp
    #         del tensor_list
            
    #         # 调用垃圾收集器
    #         gc.collect()
            
    #         # 清空缓存
    #         torch.cuda.empty_cache()
            
    #         # 同步 GPU 操作
    #         torch.cuda.synchronize()
            
    #         i += 1