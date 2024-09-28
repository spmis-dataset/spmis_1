import os
import glob
from tqdm import tqdm
from pydub import AudioSegment
import pandas as pd

def merge_audio_files_from_csv(csv_path, output_directory, log_path):
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    # 打开日志文件
    with open(log_path, 'w') as log_file:
        def log(message):
            print(message)
            log_file.write(message + '\n')

        # 读取csv文件中的所有文件路径（假设路径在第二列）
        df = pd.read_csv(csv_path)
        file_paths = df.iloc[:, 1].tolist()

        log("Initial file paths:")
        log('\n'.join(file_paths[:10]))  # 只打印前10个路径进行调试

        # 遍历每一个文件夹路径
        for folder_path in tqdm(file_paths, desc="Processing folders"):
            if not os.path.exists(folder_path):
                log(f"Folder not found: {folder_path}")
                continue

            log(f"Processing folder: {folder_path}")

            files = glob.glob(os.path.join(folder_path, "*.wav"))
            if not files:
                log(f"No .wav files found in folder: {folder_path}")
                continue

            log(f"Found {len(files)} .wav files in folder: {folder_path}")
            for file in files:
                log(f"  {file}")

            merged_files = {}

            for file in tqdm(files, desc=f"Processing files in {folder_path}", leave=False):
                try:
                    # 去掉-split{i}部分
                    parts = os.path.basename(file).split('-')
                    base_name = '-'.join(parts[:-1]) + ".wav"
                    split_index = int(parts[-1].replace('split', '').replace('.wav', ''))
                    
                    if base_name not in merged_files:
                        merged_files[base_name] = {}
                    merged_files[base_name][split_index] = file
                except Exception as e:
                    log(f"Error processing file {file}: {e}")
                    continue
            
            for base_name, split_files in tqdm(merged_files.items(), desc=f"Merging files in {folder_path}", leave=False):
                merged_file_path = os.path.join(output_directory, base_name)
                
                # 检查输出文件是否已经存在
                if os.path.exists(merged_file_path):
                    log(f"Skipping already merged file: {merged_file_path}")
                    continue

                sorted_files = [split_files[i] for i in sorted(split_files)]
                merged_audio = AudioSegment.empty()
                for file in sorted_files:
                    try:
                        audio = AudioSegment.from_wav(file)
                        merged_audio += audio
                    except Exception as e:
                        log(f"Error merging file {file}: {e}")
                        continue
                
                try:
                    merged_audio.export(merged_file_path, format="wav")
                    log(f"Merged and saved: {merged_file_path}")
                except Exception as e:
                    log(f"Error exporting merged file {merged_file_path}: {e}")

if __name__ == "__main__":
    csv_path = "retrieval_res_correct.csv"  # 修改为你的实际CSV文件路径
    output_directory = "./merged_output_wav_8_28"  # 修改为你的实际输出目录路径
    log_path = "./debug_log_all.txt"  # 日志文件路径
    merge_audio_files_from_csv(csv_path, output_directory, log_path)
