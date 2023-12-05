import glob
import os
import shutil

from TTS.vocoder.configs import ParallelWaveganConfig
from TTS.config.shared_configs import BaseAudioConfig
from TTS.vocoder.models.gan import GAN
from TTS.vocoder.datasets.preprocess import load_wav_data
from trainer import Trainer, TrainerArgs
from TTS.utils.audio import AudioProcessor

output_path = "/home/asif/tts_exp/parallelwavegan_weights_48k" #os.path.dirname(os.path.abspath(__file__))

audio_config = BaseAudioConfig(
    sample_rate=48000,
    # do_trim_silence=True,
    resample=False,
)

config = ParallelWaveganConfig(
    batch_size=64,
    eval_batch_size=16,
    num_loader_workers=0,
    num_eval_loader_workers=0,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    seq_len=2048,
    eval_split_size=1,
    print_step=1,
    print_eval=True,
    data_path="/home/asif/tts_exp/train_female/wav_48k",
    output_path=output_path,
    audio=audio_config,
)


# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

# init model
model = GAN(config, ap)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()