import time
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Process, Queue
from types import FunctionType
from typing import Dict, List, Tuple, Union
import requests

import numpy as np
import torch
import torchaudio
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    AutoModelForCausalLM,
    pipeline,
)
from transformers.utils import is_flash_attn_2_available
import json



def realtime_transcription_test(file_path: str):
    """
    Run the realtime transcription pipeline on a given audio file. This will simulate a real-time transcription from
    the given audio file. The audio file will be split into chunks of 0.5 seconds each, and each chunk will be sent to
    the pipeline.

    :param file_path: The audio file to transcribe. Should bbe in a call center format (2 channels, 16KHz).
    """
    # Load the audio file:
    audio, sample_rate = torchaudio.load(uri=file_path, format="mp3")
    audio = audio.numpy()

    # Set the sample chunk size:
    sample_chunk_size = int(sample_rate // 1.5)
    # sample_chunk_size = int(sample_rate // 150)
    audio_length = audio.shape[1]
    API_ENDPOINT = "http://localhost:8001/api/projects/default/workflows/default/infer"
    # Start the processes and transcribe the audio file:
    time.sleep(20)
    print("[System] Playing audio...")
    for i in range(0, audio_length, sample_chunk_size):
        # Split the audio into chunks and send them to the pipeline (via the requests queue):
        chunk = audio[
            :,
            i : i + sample_chunk_size
            if i + sample_chunk_size < audio_length
            else audio_length,
        ]
        audio_request = {
            "audio_chunk": chunk.tolist(),
            "timestamp": datetime.now().isoformat(),
            "flush_flag": False,
        }
        data = {"audio_request": audio_request, "question": "abs", "session_name": "1"}
        requests.post(url=API_ENDPOINT, data=json.dumps(data))

        # Wait for the next chunk as if it is a live recording:
        time.sleep(sample_chunk_size / sample_rate)
        # time.sleep(15)
    # Send the flush and last requests and wait for the processes:
    request = {
        "session_id": "1",
        "audio_chunk": [],
        "timestamp": datetime.now().isoformat(),
        "flush_flag": True,
        "question": "final message",
    }
    print("ping")
    requests.post(url=API_ENDPOINT, data=request)


def main():
    # Run the realtime transcription
    realtime_transcription_test(file_path="test.mp3")

if __name__ == "__main__":
    main()