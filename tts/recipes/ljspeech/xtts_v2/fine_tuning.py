import os
import whisper
import glob
import pandas as pd
import jaconv
from pykakasi import kakasi
from pydub import AudioSegment
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# 로깅 파라미터
RUN_NAME = "GPT_XTTS_v2.0_Japanese_FT"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# 체크포인트 저장 경로 설정
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "training")

# 학습 파라미터 설정
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True
START_WITH_EVAL = True
BATCH_SIZE = 3
GRAD_ACUMM_STEPS = 84

# 데이터셋 설정
config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="custom_japanese_dataset",
    path="C:/Users/aj200/Desktop/VS/Data/mine/MyTTSDataset/",
    meta_file_train="C:/Users/aj200/Desktop/VS/Data/mine/MyTTSDataset/metadata.csv",
    language="ja",
)

DATASETS_CONFIG_LIST = [config_dataset]

# XTTS v2.0.1 파일 다운로드 경로 설정
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

# DVAE 파일 경로 설정
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

# DVAE 파일 다운로드
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

# XTTS v2.0 체크포인트 다운로드 링크 설정
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))

# XTTS v2.0 파일 다운로드
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )

# 학습 문장 생성을 위한 스피커 참조 설정
SPEAKER_REFERENCE = [
    "C:/Users/aj200/Desktop/VS/Data/wavs/voiceFile0_0.wav",
    "C:/Users/aj200/Desktop/VS/Data/wavs/voiceFile15_0.wav",
    "C:/Users/aj200/Desktop/VS/Data/wavs/voiceFile71_0.wav",
    "C:/Users/aj200/Desktop/VS/Data/wavs/voiceFile224_0.wav",
    "C:/Users/aj200/Desktop/VS/Data/wavs/voiceFile491_0.wav"
]
LANGUAGE = config_dataset.language

# 텍스트를 '。' 기준으로 나누는 함수
def split_text(text, delimiter='。'):
    sentences = text.split(delimiter)
    return [s + delimiter for s in sentences if s]

# 오디오 파일을 분할하는 함수
def split_audio(audio_path, split_durations):
    audio = AudioSegment.from_wav(audio_path)
    segments = []
    start = 0
    for duration in split_durations:
        end = start + duration
        segments.append(audio[start:end])
        start = end
    return segments

# 텍스트와 오디오 파일을 분할하는 함수
def split_text_and_audio(sample, delimiter='。'):
    text = sample['text']
    audio_file = sample['audio_file']
    speaker_name = sample['speaker_name']
    language = sample['language']

    # 텍스트 분할
    split_texts = split_text(text, delimiter)
    
    # 오디오 길이 계산
    audio = AudioSegment.from_wav(audio_file)
    total_duration = len(audio)
    split_durations = [total_duration * len(t) / len(text) for t in split_texts]

    # 오디오 분할
    split_audios = split_audio(audio_file, split_durations)
    
    return [{'text': t, 'audio_file': audio_file.replace(".wav", f"_{i}.wav"), 'speaker_name': speaker_name, 'language': language} 
            for i, (t, a) in enumerate(zip(split_texts, split_audios))]

def main():
    # 모델 파라미터 설정
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6초
        min_conditioning_length=66150,  # 3초
        debug_loading_failures=False,
        max_wav_length=255995,  # 약 11.6초
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    # 오디오 설정 정의
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # 학습 파라미터 설정
    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="GPT XTTS training",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=0,  # 멀티프로세싱 비활성화
        num_eval_loader_workers=0,  # 멀티프로세싱 비활성화
        eval_split_max_size=256,
        print_step=100,
        plot_step=100,
        log_model_step=100,
        save_step=10000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "やあ、先生。今日はどうしたの？",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "うーん... 音楽のCDを集めるのが好きだよ。特にヘビーメタルとかね。先生も好きなものはあるの？",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "これくらいでいい？",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "なんか用？まあ、問題があったら解決してあげるけど。",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
    )

    # 설정에서 모델 초기화
    model = GPTTrainer.init_from_config(config)

    # 학습 샘플 로드 및 텍스트 분할
    try:
        train_samples, eval_samples = load_tts_samples(
            config_dataset,
            eval_split=True,
            eval_split_max_size=config.eval_split_max_size,
            eval_split_size=config.eval_split_size,
        )

        # 학습 샘플 수 확인
        if len(train_samples) == 0:
            raise ValueError("Training samples are empty.")
        if len(eval_samples) == 0:
            raise ValueError("Evaluation samples are empty.")
        
        print(f"Number of training samples: {len(train_samples)}")
        print(f"Number of evaluation samples: {len(eval_samples)}")

        # 첫 번째 학습 샘플 출력
        print(f"First training sample: {train_samples[0]}")
        print(f"First evaluation sample: {eval_samples[0]}")


        # 트레이너 초기화 및 학습 시작
        trainer = Trainer(
            TrainerArgs(
                restore_path=None,
                skip_train_epoch=False,
                start_with_eval=START_WITH_EVAL,
                grad_accum_steps=GRAD_ACUMM_STEPS,
            ),
            config,
            output_path=OUT_PATH,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
        )
        trainer.fit()
    
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()
