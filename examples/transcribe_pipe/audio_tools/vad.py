import math
from dataclasses import dataclass, field
from datetime import datetime
from types import FunctionType
from typing import Dict, List, Tuple, Union

import numpy as np
import torch

# import simpleaudio as sa

#: Number of channels expected in the audio file.
N_CHANNELS = 2


class VoiceActivityDetector:
    """
    A voice activity detection wrapper for the silero VAD model - https://github.com/snakers4/silero-vad.
    """

    def __init__(
        self,
        # Model loading kwargs:
        use_onnx: bool = False,
        # Detection kwargs:
        threshold: float = 0.5,
        sampling_rate: int = 16_000,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float("inf"),
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 1536,
        speech_pad_ms: int = 30,
        return_seconds: bool = False,
    ):
        """
        Initialize the voice activity detector.

        :param use_onnx:                Whether to use ONNX for inference.
        :param threshold:               Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
                                        probabilities ABOVE this value are considered as SPEECH. It is better to tune
                                        this parameter for each dataset separately, but "lazy" 0.5 is pretty good for
                                        most datasets.
        :param sampling_rate:           Currently, silero VAD models support 8000 and 16000 sample rates.
        :param min_speech_duration_ms:  Final speech chunks shorter min_speech_duration_ms are thrown out.
        :param max_speech_duration_s:   Maximum duration of speech chunks in seconds. Chunks longer than
                                        `max_speech_duration_s` will be split at the timestamp of the last silence that
                                        lasts more than 100ms (if any), to prevent aggressive cutting. Otherwise,
                                        they will be split aggressively just before max_speech_duration_s.
        :param min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms before
                                        separating it.
        :param window_size_samples:     Audio chunks of window_size_samples size are fed to the silero VAD model.
                                        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000
                                        sample rate and 256, 512, 768 samples for 8000 sample rate. Values other than
                                        these may affect model performance!
        :param speech_pad_ms:           Final speech chunks are padded by speech_pad_ms each side.
        :param return_seconds:          whether return timestamps in seconds (default - samples)
        """
        # Store configurations:
        self._use_onnx = use_onnx
        self._threshold = threshold
        self._sampling_rate = sampling_rate
        self._min_speech_duration_ms = min_speech_duration_ms
        self._max_speech_duration_s = max_speech_duration_s
        self._min_silence_duration_ms = min_silence_duration_ms
        self._window_size_samples = window_size_samples
        self._speech_pad_ms = speech_pad_ms
        self._return_seconds = return_seconds

        # Load the model:
        self._model: torch.Module = None
        self._get_speech_timestamps: FunctionType = None

        # Set device:
        self._torch_device: str = None

    @property
    def sampling_rate(self) -> int:
        """
        Get the sampling rate of the VAD model.

        :return: The sampling rate of the VAD model.
        """
        return self._sampling_rate

    def load(self):
        """
        Load the VAD model.
        """
        model, utils = torch.hub.load(
            # repo_or_dir="/app/models/snakers4-silero-vad-b4b6f2a",  #remote
            repo_or_dir="snakers4/silero-vad",  # local
            model="silero_vad",
            # source="local",  #remote
            force_reload=False,
            onnx=self._use_onnx,
        )
        self._model = model
        self._get_speech_timestamps = utils[0]

        # if not torch.backends.mps.is_available():
        #     if not torch.backends.mps.is_built():
        #         print("MPS not available because the current PyTorch install was not "
        #             "built with MPS enabled.")
        #     else:
        #         print("MPS not available because the current MacOS version is not 12.3+ "
        #             "and/or you do not have an MPS-enabled device on this machine.")
        #
        # else:
        #     self._torch_device = "mps"
        #     mps_device = torch.device("mps")
        #     self._model.to(mps_device)
        # (
        #     self._get_speech_timestamps,
        #     _,  # save_audio,
        #     _,  # read_audio,
        #     _,  # VADIterator,
        #     _,  # collect_chunks
        # ) = utils

    def detect_voice(
        self, audio: Union[torch.Tensor, np.ndarray, list]
    ) -> List[Dict[str, int]]:
        """
        Infer the audio thourgh the VAD model and return the speech timestamps.

        :param audio: The audio to infer. Should be a single dimension (channel) numpy array with shape (n_samples,).

        :return: The speech timestamps in the audio. A list of timestamps where each timestamp is a dictionary with the
                 following keys:

                 * "start": The start sample index of the speech in the audio.
                 * "end":   The end sample index of the speech in the audio.
        """
        # Cast to a torch tensor:
        if isinstance(audio, list):
            audio = torch.tensor(audio, dtype=torch.float32)
        elif isinstance(audio, np.ndarray):
            audio = torch.tensor(audio, dtype=torch.float32)
        if self._torch_device is not None:
            audio = audio.to(self._torch_device)
        # Detect speech:
        speech_timestamps = self._get_speech_timestamps(
            audio,
            self._model,
            threshold=self._threshold,
            min_speech_duration_ms=self._min_speech_duration_ms,
            max_speech_duration_s=self._max_speech_duration_s,
            min_silence_duration_ms=self._min_silence_duration_ms,
            speech_pad_ms=self._speech_pad_ms,
            sampling_rate=self._sampling_rate,
            window_size_samples=self._window_size_samples,
        )

        return speech_timestamps


class VADBufferPerChannel:
    # TODO: Inherit from `mlrun.serving.V2ModelServer` once moving to production.
    """
    A per channel VAD buffer class. This class will buffer the audio chunks per channel, and will flush the buffer when
    needed. The buffer will be flushed when one of the following conditions is met:

      1. The flush flag is True.
      2. There was no speech detected and the buffer is not empty (equal to the current samples).
      3. The buffer's length (in seconds) is bigger than the set maximum.
    """

    @dataclass
    class _ChannelBuffer:
        request_timestamp: datetime = field(default=None)
        first_voice_detection_timestamp: int = field(default=None)
        buffer: np.ndarray = field(default_factory=lambda: np.array([]))
        last_samples: np.ndarray = field(default_factory=lambda: np.array([]))
        is_concluded: bool = field(default=False)

        def __lt__(self, other: "VADBufferPerChannel._ChannelBuffer"):
            if self.request_timestamp == other.request_timestamp:
                if (
                    self.first_voice_detection_timestamp is not None
                    and other.first_voice_detection_timestamp is not None
                ):
                    return (
                        self.first_voice_detection_timestamp
                        <= other.first_voice_detection_timestamp
                    )
                return True
            return self.request_timestamp <= other.request_timestamp

        def reset(self):
            self.request_timestamp = None
            self.first_voice_detection_timestamp = None
            self.buffer = self.last_samples
            self.last_samples = np.array([])
            self.is_concluded = False

        def to_dict(self, json_serializable: bool = False) -> dict:
            return {
                "timestamp": (
                    self.request_timestamp.isoformat()
                    if json_serializable
                    else self.request_timestamp
                ),
                "buffer": self.buffer.tolist() if json_serializable else self.buffer,
                "status": "complete" if self.is_concluded else "intermediate",
            }

        @classmethod
        def from_dict(cls, data: dict):
            buffer = cls()
            buffer.request_timestamp = datetime.fromisoformat(data["timestamp"])
            buffer.buffer = np.array(data["buffer"])
            buffer.is_concluded = data["status"] == "complete"
            return buffer

        def is_voice_detected(self):
            return self.first_voice_detection_timestamp is not None

    def __init__(
        self,
        n_channels: int,
        vad_kwargs: dict = None,
        sampling_rate: int = 16_000,
        max_buffer_size_s: float = None,
    ):
        """
        Initialize the per channel VAD buffer class. This class will buffer the audio chunks per channel, and will flush

        :param n_channels:          Number of channels expected in the audio file.
        :param vad_kwargs:          Keyword arguments to pass to the `VoiceActivityDetector` class.
        :param sampling_rate:       Sampling rate of the expected audio to detect voice in.
        :param max_buffer_size_s:   The maximum buffer size in seconds. If the buffer's length (in seconds) is bigger
        """
        vad_kwargs = vad_kwargs or {}
        vad_kwargs["sampling_rate"] = sampling_rate
        self._vad = VoiceActivityDetector(**(vad_kwargs or {}))

        self._n_channels = n_channels
        self._max_buffer_size_s = max_buffer_size_s

        self._sessions: Dict[str, List[VADBufferPerChannel._ChannelBuffer]] = {}

        self._vad.load()

    def __call__(self, request: dict, buffers=None) -> tuple:
        """
        Process the request through the VAD and return the buffer collected when a silence was detected (or when maximum
        buffer length reached).

        :param request: The request to process. The request should be in the following format::

                        {
                            "session_id": "1",
                            "audio": [0.1, 0.2, ...],
                            "timestamp": "2021-08-16T12:00:00",
                            "flush_flag": False,
                        }

        :return: The buffer collected when a silence was detected (or when maximum buffer length reached) in the
                 following format::

                 {
                     "session_id": "1",
                     "audio_channels": {
                         0: {
                             "timestamp": "2021-08-16T12:00:00",
                             "buffer": [0.1, 0.2, ...],
                         },
                         1: {
                             "timestamp": "2021-08-16T12:00:00",
                             "buffer": [0.3, 0.4, ...],
                         },
                     },
                 }
        """
        # Unpack request:
        audio: Union[torch.Tensor, np.ndarray, list] = request["audio_chunk"]
        timestamp: Union[str, datetime] = request["timestamp"]
        flush_flag: bool = request.get("flush_flag", False)

        # Parse the timestamp:
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        # Cast to a numpy array:
        if isinstance(audio, list):
            audio = np.array(audio, dtype=np.float32)
        elif isinstance(audio, torch.Tensor):
            audio = audio.numpy()

        # Add channel dimension if needed:
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=0)

        # Get the session buffers (create one if not exists):
        if not buffers:
            buffers = [
                VADBufferPerChannel._ChannelBuffer() for _ in range(self._n_channels)
            ]
        else:
            buffers = [
                VADBufferPerChannel._ChannelBuffer.from_dict(data)
                for data in buffers.values()
            ]

        # Detect speech:
        transcription_requests: List[
            Tuple[int, VADBufferPerChannel._ChannelBuffer]
        ] = []
        for channel, samples in enumerate(audio):
            # Set the channel's timestamp if not exists:
            if buffers[channel].request_timestamp is None:
                buffers[channel].request_timestamp = timestamp
            # Detect speech on the new chunk's samples:
            speech_timestamps = self._vad.detect_voice(audio=samples)

            # If there are speech timestamps, update the channel's first voice detection timestamp if not exist (means
            # it's a first voice detection in the channel):
            if speech_timestamps and not buffers[channel].is_voice_detected():
                buffers[channel].first_voice_detection_timestamp = speech_timestamps[0][
                    "start"
                ]

            buffer_len_before_concat = (
                buffers[channel].buffer.size / self._vad.sampling_rate
            )

            # Concatenate the samples to the channel's buffer:
            if not buffers[channel].is_voice_detected() and not speech_timestamps:
                buffers[channel].buffer = samples
            else:
                buffers[channel].buffer = np.concatenate(
                    (buffers[channel].buffer, samples), axis=0
                )

            # Used later for finding out whether a new second has passed
            buffer_len_after_concat = (
                buffers[channel].buffer.size / self._vad.sampling_rate
            )

            # Check if needed to flush the buffer, the buffer will be flushed if there was a detected voice and one of
            # the following conditions is met:
            #  1. The flush flag is True.
            #  2. There was no speech detected and the buffer is not empty (equal to the current samples).
            #  3. The buffer's length (in seconds) is bigger than the set maximum.

            # Detect speech on the new chunk's samples:
            lookback_index = min(buffers[channel].buffer.size, samples.size * 3)
            most_recent_audio = buffers[channel].buffer[-lookback_index:]
            speech_timestamps_new = self._vad.detect_voice(audio=most_recent_audio)
            if buffers[channel].is_voice_detected() and (
                # 1.
                flush_flag
                or
                # 2.
                not speech_timestamps_new
                or
                # 3.
                (
                    self._max_buffer_size_s is not None
                    and buffers[channel].buffer.size / self._vad.sampling_rate
                    >= self._max_buffer_size_s
                )
            ):
                transcription_requests.append((channel, buffers[channel]))
                buffers[channel].is_concluded = True
                buffers[channel].last_samples = np.array([])
                print("transcription_requests:", transcription_requests)
            else:
                # Check if buffer has surpassed a new second
                if (
                    math.floor(buffer_len_after_concat)
                    - math.floor(buffer_len_before_concat)
                    == 1
                ):
                    transcription_requests.append((channel, buffers[channel]))
                    print("transcription_requests:", transcription_requests)

        # Sort the transcription requests by timestamps:
        transcription_requests.sort(key=lambda x: x[1])

        serialized_buffers = {}
        for channel, buffer in enumerate(buffers):
            # Add the channel's buffer to the response:
            serialized_buffers[channel] = buffer.to_dict(json_serializable=True)
            # Reset the buffer:
            if buffer.is_concluded:
                buffer.reset()
        # Return the response to transcribe:
        return transcription_requests, serialized_buffers
