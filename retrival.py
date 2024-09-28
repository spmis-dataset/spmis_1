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
import os
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

device = torch.device('cpu')
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').to(device)

def extract_feature(tsrs, path=None):
    # data, samplerate = sf.read(path)
    if os.path.exists(path.replace('.wav', '-sv.pt')):
        return torch.load(path.replace('.wav', '-sv.pt'), map_location=device)
    data = feature_extractor(tsrs, sampling_rate=16000,padding=False, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model(**data).embeddings
    torch.save(embeddings, path.replace('.wav', '-sv.pt'))
    return embeddings




def get_index(path):
    index = faiss.read_index(path + 'names.index', faiss.IO_FLAG_MMAP)
    names = torch.load(path + 'index_name')
    res = faiss.StandardGpuResources() 
    index = faiss.index_cpu_to_gpu(res, 7, index)  # 0 表示使用第一块GPU
    return names, index


def normalize_L2(query):
    # 计算每个向量的 L2 范数
    norms = torch.norm(query, p=2, dim=1, keepdim=True)
    # 将每个向量除以其 L2 范数
    normalized_query = query / norms
    return normalized_query

# res = faiss.StandardGpuResources()
def retrieve_data(df, names, index):
    data_res = {
        'test_speaker': [],
        'similar_speaker': [],
        'distance': [],
    }
    # index = faiss.index_cpu_to_gpu(res, 4, index)
    for path in tqdm(df['path']):
        files = os.listdir(path)
        pt_files = [w for w in files if w.endswith('.wav')]
        # query = [extract_feature(sf.read(path + '/' + p)[0], path + '/' + p) for p in pt_files]
        # query = pt_files
        # query = []
        # for pt in pt_files:
        #     tmp_data = torch.load(path + '/' + pt, map_location=device)
        #     query.append(tmp_data)
        
        query = torch.mean(torch.cat([extract_feature(sf.read(path + '/' + p)[0], path + '/' + p) for p in pt_files], dim = 0), dim = 0, keepdim=True)
        # query = torch.mean(query, dim = 0, keepdim=True)
        # query = query.numpy()
        # query = torch.nn.functional.normalize(query, dim=-1)
        query = normalize_L2(query)
        # faiss.normalize_L2(query)
        
        D, I = index.search(query, k=1)
        
        # try:
        # idx = I.item()
        idx = I.tolist()[0]
        # while idx > len(names) or idx < 0:
        #     D, I = index.search(query, k=3)
        #     # idx = I.item()
        #     idx = I.tolist()
        dis = D.tolist()[0]
        for i in range(len(idx)):
            data_res['test_speaker'].append(path)
        # tmp_name = '-'.join([str(names[i] for i in idx[0]])
            data_res['similar_speaker'].append(names[idx[i]])
        # except:
        #     print(len(data_res['similar_speaker']))
        #     print(names[I.item()])
        #     exit()
            data_res['distance'].append(dis[i])
    save_file = pd.DataFrame.from_dict(data_res)
    save_file.to_csv('/mnt/data1/liupeizhuo/detect/final_new/150_cosine_va_clean/retrieval_res_cos.csv')
    print('Saved!')


def main():
    root = '/mnt/data1/liupeizhuo/detect/final_new/'
    df = pd.read_csv(root + '8_24_misinformation.csv')
    names, index = get_index(root + '150_cosine_va_clean/')
    retrieve_data(df, names, index)

main()