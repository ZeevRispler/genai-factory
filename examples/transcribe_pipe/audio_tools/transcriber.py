from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Literal, Union

import numpy as np
import torch
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperModel,
    WhisperProcessor,
    pipeline,
)
from scipy.io import wavfile
import tempfile
from pathlib import Path
from typing import Union
# import simpleaudio as sa

#: Number of channels expected in the audio file.
N_CHANNELS = 2


class Transcriber:
    """
    A transcription wrapper for the Huggingface's ASR pipeline -
    https://huggingface.co/transformers/main_classes/pipelines.html#transformers.AutomaticSpeechRecognitionPipeline to
    use with OpenAI's Whisper models - https://huggingface.co/openai.
    """

    def __init__(
        self,
        # Model loading kwargs:
        model_name: str = "openai/whisper-tiny",  # local
        device: str = None,
        use_flash_attention_2: bool = False,
        use_better_transformers: bool = False,
        # Generation kwargs:
        max_new_tokens: int = 128,
        chunk_length_s: int = 30,
        batch_size: int = 16,
        return_timestamps: Union[bool, Literal["words"]] = False,
        spoken_language: str = None,
        translate_to_english: bool = False,
    ):
        """
        Initialize the transcriber.

        :param model_name:              The model name to use. Should be a model from the OpenAI's Whisper models for
                                        best results (for example "tiny", "base", "large", etc.).
        :param device:                  The device to use for inference. If not given, will use GPU if available.
        :param use_flash_attention_2:   Whether to use the Flash Attention 2 implementation. It can be used only with
                                        one of the following GPUs: Nvidia H series and Nvidia A series. T4 support will
                                        be available soon.
        :param use_better_transformers: Whether to use the Better Transformers library to further optimize the model.
                                        Should be used for all use cases that do not support flash attention 2.
        :param max_new_tokens:          The maximum number of new tokens to generate. This is used to limit the
                                        generation length.
        :param chunk_length_s:          The audio chunk to use in seconds.
        :param batch_size:              The batch size to use for inference.
        :param return_timestamps:       Whether to return the timestamps of the transcriptions. If "words", will return
                                        the timestamps of each word in the transcription.
        :param spoken_language:         Aim whisper to know what language is spoken. If None, it will try to detect it.
        :param translate_to_english:    Whether to translate the transcriptions to English.
        """
        # Store the model name:
        self._model_name = model_name

        # Store loading configurations:
        self._device = device
        self._use_flash_attention_2 = use_flash_attention_2
        self._use_better_transformers = use_better_transformers

        # Store generation configurations:
        self._max_new_tokens = max_new_tokens
        self._chunk_length_s = chunk_length_s
        self._batch_size = batch_size
        self._return_timestamps = return_timestamps
        self._spoken_language = spoken_language
        self._translate_to_english = translate_to_english

        # Prepare pipeline and generation kwargs class variables:
        # TODO: Replace the `AutomaticSpeechRecognitionPipeline` with a custom class that will handle batches and
        #       per channel prompts for inference.
        self._model: WhisperModel = None
        self._processor: WhisperProcessor = None
        self._pipeline: AutomaticSpeechRecognitionPipeline = None
        self._pipeline_kwargs: dict = None
        self._generate_kwargs: dict = None

    def load(self):
        """
        Load the transcriber.
        """
        # Set the device and data type to use (prefer GPU if available):
        # device = self._device or "cuda" if torch.cuda.is_available() else "cpu"
        device = (
            self._device or "cuda" if torch.cuda.is_available() else "mps"
        )  # mps for macs
        torch_dtype = torch.float16 if device != "cpu" else torch.float32

        # Download the model and set it up in memory:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            token="...",
            pretrained_model_name_or_path=self._model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="flash_attention_2"
            if self._use_flash_attention_2
            else "sdpa",
        )
        if self._use_better_transformers:
            model = model.to_bettertransformer()

        # Move model to GPU (if available):
        model.to(device)

        # Get the processor:
        processor = AutoProcessor.from_pretrained(self._model_name)

        # Store the model and processor:
        self._model = model
        self._processor = processor

        # Initialize the speech recognition pipeline:
        self._pipeline = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=self._max_new_tokens,
            chunk_length_s=self._chunk_length_s,
            batch_size=self._batch_size,
            return_timestamps=self._return_timestamps,
            torch_dtype=torch_dtype,
            device=device,
        )

        # Prepare the generation kwargs:
        self._pipeline_kwargs = {
            "chunk_length_s": self._chunk_length_s,
            "batch_size": self._batch_size,
            "return_timestamps": self._return_timestamps,
        }
        self._generate_kwargs = {
            "language": self._spoken_language,
            "task": "translate" if self._translate_to_english else "transcribe",
        }
        print("transcription model loaded")

    def transcribe(
        self,
        audio: Union[torch.Tensor, np.ndarray, list],
        sampling_rate: int,
        # prompt: str = ""  # TODO: Uncomment when the prompting is handled correctly
    ) -> dict:
        """
        Transcribe the given audio.

        :param audio:         The audio to transcribe. Should be a single dimension (channel) numpy array with shape
                              (n_samples,).
        :param sampling_rate: The sampling rate of the audio.

        :return: The transcription of the audio (look at the ASR pipeline outputs to know more).
        """
        # Cast to a numpy array:
        if isinstance(audio, list):
            audio = np.array(audio, dtype=np.float32)
        elif isinstance(audio, torch.Tensor):
            audio = audio.numpy()

        # Check audio length
        audio_length_seconds = len(audio) / sampling_rate
        if audio_length_seconds < 0.1:
            return {"text": ""}

        # Copy the generation kwargs:
        generate_kwargs = self._generate_kwargs.copy()

        self._pipeline.model.config.forced_decoder_ids = (
            self._processor.get_decoder_prompt_ids(**generate_kwargs)
        )

        # TODO: This is currently very naively implemented in Huggingface's pipeline, so we need to handle it manually
        #       when we develop our own pipeline.
        # if prompt:
        #     prompt_ids = self._processor.get_prompt_ids(text=prompt)
        #     generate_kwargs["prompt_ids"] = prompt_ids
        # Infer through the pipeline:

        result = self._pipeline(
            {"raw": audio, "sampling_rate": sampling_rate},
            **self._pipeline_kwargs,
        )
        return result


class ApiTranscriber:
    def __init__(
            self,
            model_name: str = "whisper-1",
            max_new_tokens: int = 128,
            chunk_length_s: int = 30,
            batch_size: int = 16,
            return_timestamps: Union[bool, Literal["words"]] = False,
            spoken_language: str = None,
            translate_to_english: bool = False
    ):
        from openai import OpenAI
        self.model = OpenAI()
        # Store generation configurations:
        self._max_new_tokens = max_new_tokens
        self._chunk_length_s = chunk_length_s
        self._batch_size = batch_size
        self._return_timestamps = return_timestamps
        self._spoken_language = spoken_language
        self._translate_to_english = translate_to_english
        self._model_name = model_name

    def load(self):
        pass

    def transcribe(self, audio: Union[torch.Tensor, np.ndarray, list], sampling_rate: int, **generation_kwargs) -> dict:
        """audio.transcriptions.create(
        Convert audio data to a temporary WAV file, transcribe with Whisper, and cleanup.

        Parameters:
        audio: Audio data as torch.Tensor, np.ndarray, or list
        sampling_rate: Sampling rate in Hz

        Returns:
        dict: Whisper transcription result
        """
        # Convert input to numpy array if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        elif isinstance(audio, list):
            audio = np.array(audio)

        # Check audio length
        audio_length_seconds = len(audio) / sampling_rate
        if audio_length_seconds < 0.1:
            return {"text": ""}

        # Convert float arrays to 16-bit PCM
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio = np.clip(audio, -1, 1)
            audio = (audio * 32767).astype(np.int16)

        try:
            # Create temporary directory and file
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "temp_audio.wav"

                # Save audio file
                wavfile.write(temp_path, sampling_rate, audio)

                # open file:
                audio_file = open(temp_path, "rb")

                # Transcribe using the model stored in self
                result = self.model.audio.transcriptions.create(
                    file=audio_file,
                    model=self._model_name,
                    **generation_kwargs
                          )

                return result.to_dict()

        except Exception as e:
            raise Exception(f"Error during transcription process: {str(e)}")


class TranscriberPerChannel:
    # TODO: Inherit from `mlrun.serving.V2ModelServer` once moving to production.
    """
    A per channel transcriber class. This class will transcribe each channel separately, and will return the
    transcriptions per channel.
    """

    @dataclass
    class _ChannelTranscription:
        timestamp: datetime
        text: str

        def to_dict(self, json_serializable: bool = False) -> dict:
            return {
                "timestamp": (
                    self.timestamp.isoformat() if json_serializable else self.timestamp
                ),
                "text": self.text,
            }

    def __init__(
        self,
        n_channels: int,
        transcriber_kwargs: dict = None,
        sampling_rate: int = 16_000,
        api: bool = False,
    ):
        """
        Initialize the per channel transcriber class.

        :param n_channels:         Number of channels expected in the audio file.
        :param transcriber_kwargs: Keyword arguments to pass to the `Transcriber` class.
        :param sampling_rate:      The sampling rate of the audio to transcribe.
        """
        # Store configurations:
        self._n_channels = n_channels
        self._sampling_rate = sampling_rate

        # Initialize the transcriber and load it:
        self._transcriber = Transcriber(**(transcriber_kwargs or {})) if not api else ApiTranscriber(**(transcriber_kwargs or {}))
        self._transcriber.load()


    def __call__(self, request: dict) -> dict:
        """
        Process the request through the transcriber and return the transcriptions per channel.

        :param request: The request to process. The request should be in the following format::

                        {
                            0: {
                                "timestamp": "2021-08-16T12:00:00",
                                "buffer": [0.1, 0.2, ...],
                            },
                            1: {
                                "timestamp": "2021-08-16T12:00:00",
                                "buffer": [0.3, 0.4, ...],
                        },
                        }

        :return: The transcriptions per channel in the following format::

                        {
                            0: {
                                "timestamp": "2021-08-16T12:00:00",
                                "text": "Hello world!",
                            },
                            1: {
                                "timestamp": "2021-08-16T12:00:00",
                                "text": "Hello Transcriber!",
                            },
                        }
        """
        # Unpack request:
        transcription_requests: Dict[int, dict] = request

        response: Dict[int, dict] = {}

        for channel, channel_buffer in transcription_requests:
            # Unpack channel buffer:
            timestamp: Union[str, datetime] = channel_buffer.request_timestamp
            if not timestamp:
                continue
            buffer: Union[torch.Tensor, np.ndarray, list] = channel_buffer.buffer
            # Parse the timestamp:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            # Cast to a numpy array:
            if isinstance(buffer, list):
                buffer = np.array(buffer, dtype=np.float32)
            elif isinstance(buffer, torch.Tensor):
                buffer = buffer.numpy()
            # Transcribe the buffer:
            transcription = self._transcriber.transcribe(
                audio=buffer,
                sampling_rate=self._sampling_rate,
                # prompt=prompts[channel]  # TODO: Uncomment when the prompting is handled correctly
            )["text"]

            # TODO: Uncomment when the prompting is handled correctly
            # # Remove prompt from output:
            # if prompts[channel] and transcription.startswith(prompts[channel]):
            #     transcription = transcription[len(prompts[channel]):]
            # Update the channel's prompt:
            # prompts[channel] = transcription
            # Collect the response:
            response[channel] = TranscriberPerChannel._ChannelTranscription(
                timestamp=timestamp, text=transcription,
            ).to_dict(json_serializable=True)

        # Return the response to transcribe:
        return response
