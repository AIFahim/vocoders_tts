import os

from trainer import Trainer, TrainerArgs

from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import HifiganConfig
from TTS.config.shared_configs import BaseAudioConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

output_path = "/home/asif/tts_exp/hifigan_weights_48k" #os.path.dirname(os.path.abspath(__file__))
audio_config = BaseAudioConfig(
    sample_rate=48000,
    # do_trim_silence=True,
    resample=False,
)



config = HifiganConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=5,
    epochs=1000,
    seq_len=8192,
    pad_short=2000,
    use_noise_augment=False,
    eval_split_size=10,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    lr_gen=1e-4,
    lr_disc=1e-4,
    data_path= "/home/asif/tts_exp/train_female/wav_48k", #os.path.join(output_path, "../LJSpeech-1.1/wavs/"),
    output_path=output_path,
    audio=audio_config,
    l1_spec_loss_params={'use_mel': True, 'sample_rate': audio_config.sample_rate, 'n_fft': 1024, 'hop_length': 256, 'win_length': 1024, 'n_mels': 80, 'mel_fmin': 0.0, 'mel_fmax': None}
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