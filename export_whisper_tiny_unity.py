#!/usr/bin/env python
# coding: utf-8

"""
一つのPythonスクリプトで、
 - openai/whisper-tiny (多言語版) をHugging Faceからロード
 - logmel_spectrogram, encoder_model, decoder_model, decoder_with_past_model
   の4つのONNXファイルを出力
 を行うサンプル。
 
実行例:
  python export_whisper_tiny_unity.py
 成功すればカレントディレクトリに:
   - logmel_spectrogram.onnx
   - encoder_model.onnx
   - decoder_model.onnx
   - decoder_with_past_model.onnx
 が出力されます。
"""

import os
import torch
import torch.nn as nn
import librosa
import numpy as np

from transformers import WhisperForConditionalGeneration


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        n_mels=80,
        n_fft=400,
        hop_length=160,
        sample_rate=16000,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        # n_fft//2 + 1 = freq_bins
        freq_bins = self.n_fft // 2 + 1

        # メルフィルタ: shape = (n_mels, freq_bins)
        mel_filter = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=0,
            fmax=self.sample_rate/2,
        )
        # 上記は (n_mels, freq_bins)
        # 転置して (freq_bins, n_mels) にしておくと行列積が便利
        mel_filter = torch.from_numpy(mel_filter).float().transpose(0, 1)
        self.register_buffer("mel_filter", mel_filter)  # shape: (freq_bins, n_mels)

        # ハニング窓
        window = torch.hann_window(self.n_fft)
        self.register_buffer("window", window)

    def forward(self, wav: torch.Tensor):
        """
        wav shape: (batch, samples)
        return: log_mel shape: (batch, n_mels, time)
        """
        # STFT: return_complex=False => 出力 (batch, freq_bins, frames, 2)
        stft_out = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            pad_mode='reflect',
            return_complex=False,
        )
        # => stft_out.shape: (batch, freq_bins, frames, 2)
        # 実部＆虚部
        real = stft_out[..., 0]  # (batch, freq_bins, frames)
        imag = stft_out[..., 1]  # (batch, freq_bins, frames)

        # 振幅スペクトル = sqrt(real^2 + imag^2)
        magnitudes = (real**2 + imag**2).sqrt()  # shape: (batch, freq_bins, frames)

        # mel_filter.shape = (freq_bins, n_mels)
        # => 行列積には “(batch, frames, freq_bins) x (freq_bins, n_mels)” = (batch, frames, n_mels)
        magnitudes_T = magnitudes.transpose(1, 2)  # (batch, frames, freq_bins)

        mel = torch.matmul(magnitudes_T, self.mel_filter)  # => (batch, frames, n_mels)
        mel = mel.transpose(1, 2)  # => (batch, n_mels, frames)

        # log変換
        log_mel = torch.clamp(mel, min=1e-10).log10()
        return log_mel


class WhisperEncoder(nn.Module):
    def __init__(self, hf_whisper: WhisperForConditionalGeneration):
        super().__init__()
        # HFの内部モデル: whisper.model.encoder
        self.encoder = hf_whisper.model.encoder

    def forward(self, mel_input):
        """
        mel_input: (batch, 80, time)
        returns: (batch, time, hidden_size)
        """
        encoder_outputs = self.encoder(
            mel_input,
            output_attentions=False,
            output_hidden_states=False,
        )
        return encoder_outputs.last_hidden_state


class WhisperDecoderNoPast(nn.Module):
    """
    デコーダをpastなしで呼び出し、present_key_valuesを出力する。
    """
    def __init__(self, hf_whisper: WhisperForConditionalGeneration):
        super().__init__()
        self.decoder = hf_whisper.model.decoder
        self.lm_head = hf_whisper.proj_out  # Whisper用最終出力層

    def forward(self, input_ids, encoder_hidden_states):
        # past_key_values = None, use_cache=True で呼び出し
        outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=None,
            use_cache=True
        )
        hidden = outputs.last_hidden_state  # (batch, dec_seq, hidden_size)
        logits = self.lm_head(hidden)       # (batch, dec_seq, vocab_size)

        pkv = outputs.past_key_values  # tuple of (dec_k, dec_v, enc_k, enc_v) × num_layers

        out_dict = {"logits": logits}
        for i, (dec_k, dec_v, enc_k, enc_v) in enumerate(pkv):
            out_dict[f"present.{i}.decoder.key"]   = dec_k
            out_dict[f"present.{i}.decoder.value"] = dec_v
            out_dict[f"present.{i}.encoder.key"]   = enc_k
            out_dict[f"present.{i}.encoder.value"] = enc_v

        return out_dict


class WhisperDecoderWithPast(nn.Module):
    def __init__(self, hf_whisper: WhisperForConditionalGeneration, num_layers=4):
        super().__init__()
        self.decoder = hf_whisper.model.decoder
        self.lm_head = hf_whisper.proj_out
        self.num_layers = num_layers

    def forward(
        self,
        input_ids,               # shape: (batch, dec_seq)
        encoder_hidden_states,   # shape: (batch, enc_seq, hidden_size)
        *past_key_values_flat
    ):
        # past_kvsを再構築
        pkv_tuple = []
        for i in range(self.num_layers):
            offset = i * 4
            dec_k = past_key_values_flat[offset + 0]
            dec_v = past_key_values_flat[offset + 1]
            enc_k = past_key_values_flat[offset + 2]
            enc_v = past_key_values_flat[offset + 3]
            pkv_tuple.append((dec_k, dec_v, enc_k, enc_v))

        outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=tuple(pkv_tuple),
            use_cache=True
        )
        hidden = outputs.last_hidden_state
        logits = self.lm_head(hidden)

        pkv = outputs.past_key_values
        out_dict = {"logits": logits}
        for i, (dec_k, dec_v, enc_k, enc_v) in enumerate(pkv):
            out_dict[f"present.{i}.decoder.key"]   = dec_k
            out_dict[f"present.{i}.decoder.value"] = dec_v
            out_dict[f"present.{i}.encoder.key"]   = enc_k
            out_dict[f"present.{i}.encoder.value"] = enc_v

        return out_dict


def main():
    print("Loading Hugging Face model: openai/whisper-tiny (multilingual)")
    hf_whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    hf_whisper.eval()

    # 1) logmel_spectrogram
    print("Exporting logmel_spectrogram.onnx ...")
    logmel = LogMelSpectrogram()
    logmel.eval()

    dummy_wav = torch.randn(1, 16000 * 30)
    torch.onnx.export(
        logmel,
        dummy_wav,
        "logmel_spectrogram.onnx",
        opset_version=17,
        input_names=["audio_signal"],
        output_names=["logmel"],
        dynamic_axes={
            "audio_signal": {1: "samples"},
            "logmel": {2: "time"}
        }
    )

    # 2) encoder_model
    print("Exporting encoder_model.onnx ...")
    encoder_model = WhisperEncoder(hf_whisper)
    encoder_model.eval()
    dummy_logmel = torch.randn(1, 80, 3000)
    torch.onnx.export(
        encoder_model,
        dummy_logmel,
        "encoder_model.onnx",
        opset_version=17,
        input_names=["mel"],
        output_names=["encoder_hidden_states"],
        dynamic_axes={
            "mel": {2: "time"},
            "encoder_hidden_states": {1: "time"}
        }
    )

    # 3) decoder_model (NoPast)
    print("Exporting decoder_model.onnx ...")
    decoder_no_past = WhisperDecoderNoPast(hf_whisper)
    decoder_no_past.eval()
    dummy_input_ids = torch.randint(0, 50000, (1, 5))
    dummy_enc_out   = torch.randn(1, 3000, 384)
    torch.onnx.export(
        decoder_no_past,
        (dummy_input_ids, dummy_enc_out),
        "decoder_model.onnx",
        opset_version=17,
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=(
            ["logits"] +
            [f"present.{i}.decoder.key" for i in range(4)] +
            [f"present.{i}.decoder.value" for i in range(4)] +
            [f"present.{i}.encoder.key" for i in range(4)] +
            [f"present.{i}.encoder.value" for i in range(4)]
        ),
        dynamic_axes={
            "input_ids": {1: "dec_seq"},
            "encoder_hidden_states": {1: "enc_seq"},
            "logits": {1: "dec_seq"},
        }
    )

    # 4) decoder_with_past_model
    print("Exporting decoder_with_past_model.onnx ...")
    decoder_with_past = WhisperDecoderWithPast(hf_whisper, num_layers=4)
    decoder_with_past.eval()
    dummy_input_ids_2 = torch.randint(0, 50000, (1, 1))
    dummy_enc_out_2   = torch.randn(1, 3000, 384)

    # 過去のkv(4レイヤー) => (dec_k, dec_v, enc_k, enc_v)*4=16個
    dummy_past_kvs = []
    for i in range(4):
        dec_k = torch.randn(1, 6, 5, 64)  # (batch=1, heads=6, past_dec_seq=5, head_dim=64)
        dec_v = torch.randn(1, 6, 5, 64)
        enc_k = torch.randn(1, 6, 3000, 64)
        enc_v = torch.randn(1, 6, 3000, 64)
        dummy_past_kvs += [dec_k, dec_v, enc_k, enc_v]

    input_names = ["input_ids", "encoder_hidden_states"]
    for i in range(4):
        input_names += [
            f"past_key_values.{i}.decoder.key",
            f"past_key_values.{i}.decoder.value",
            f"past_key_values.{i}.encoder.key",
            f"past_key_values.{i}.encoder.value",
        ]

    output_names = ["logits"]
    for i in range(4):
        output_names += [
            f"present.{i}.decoder.key",
            f"present.{i}.decoder.value",
            f"present.{i}.encoder.key",
            f"present.{i}.encoder.value",
        ]

    # ★ここで past_key_values の軸を動的指定する
    #  - decoder.key/value は shape = (batch, heads, past_dec_seq, head_dim)
    #  - encoder.key/value は shape = (batch, heads, enc_seq, head_dim)
    #   などに合わせて {2: "past_dec_seq"} or {2: "enc_seq"} を指定します
    dynamic_axes_dict = {
        "input_ids": {1: "dec_seq"},
        "encoder_hidden_states": {1: "enc_seq"},
        "logits": {1: "dec_seq"},
    }
    for layer_idx in range(4):
        # decoder key/value => dimension 2 = past_dec_seq
        dynamic_axes_dict[f"past_key_values.{layer_idx}.decoder.key"] = {2: "past_dec_seq"}
        dynamic_axes_dict[f"past_key_values.{layer_idx}.decoder.value"] = {2: "past_dec_seq"}
        # encoder side => dimension 2 = enc_seq (if we want dynamic, same as "encoder_hidden_states")
        dynamic_axes_dict[f"past_key_values.{layer_idx}.encoder.key"] = {2: "enc_seq"}
        dynamic_axes_dict[f"past_key_values.{layer_idx}.encoder.value"] = {2: "enc_seq"}

    torch.onnx.export(
        decoder_with_past,
        (dummy_input_ids_2, dummy_enc_out_2, *dummy_past_kvs),
        "decoder_with_past_model.onnx",
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict
    )

    print("All 4 ONNX files have been exported successfully!")


if __name__ == "__main__":
    main()
