import os
import glob
import whisper
from tqdm import tqdm

# 加载Whisper模型
model = whisper.load_model("medium").to("cuda")

def transcribe_audio(audio_path, language="en"):
    # 加载和处理音频文件
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # 生成log-Mel spectrogram
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # 检测语言
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # 解码音频
    options = whisper.DecodingOptions(language=language)
    result = model.decode(mel, options)
    return result.text

def transcribe_merged_audio_files(input_directory, output_directory):
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    # 获取输入目录中所有的 .wav 文件
    audio_files = glob.glob(os.path.join(input_directory, "*.wav"))
    print(f"Found {len(audio_files)} audio files in {input_directory}")

    # 遍历所有音频文件
    for audio_path in tqdm(audio_files, desc="Transcribing audio files"):
        base_name = os.path.basename(audio_path)
        output_file_path = os.path.join(output_directory, base_name.replace('.wav', '.txt'))
        
        # 检查输出文件是否已经存在
        if os.path.exists(output_file_path):
            print(f"Skipping already transcribed file: {output_file_path}")
            continue

        print(f"Transcribing audio file: {audio_path}")
        transcription = transcribe_audio(audio_path, language="en")
        
        # 保存转录结果到输出文件
        with open(output_file_path, 'w') as output_file:
            output_file.write(transcription + "\n")
        print(f"Transcribed and saved: {audio_path} to {output_file_path}")

if __name__ == "__main__":
    input_directory = "./merged_output_wav"  # 修改为你的实际输入目录路径
    output_directory = "./transcriptions_out"  # 修改为你的实际输出目录路径
    transcribe_merged_audio_files(input_directory, output_directory)