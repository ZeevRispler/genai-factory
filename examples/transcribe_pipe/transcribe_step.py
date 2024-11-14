# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional
from logging import getLogger

from genai_factory.chains.base import ChainRunner
from audio_tools import TranscriberPerChannel, N_CHANNELS

logger = getLogger(__name__)


class Transcribe(ChainRunner):
    """
    A chain runner that transcribes audio to text.
    Supports multi-channel audio with separate transcription for agent and client channels.
    """

    DEFAULT_SPEAKER_MAP = {
        0: "Agent",
        1: "User"
    }

    def __init__(
            self,
            model: str = "openai/whisper-tiny",
            use_gpu: bool = False,
            device_type: str = "cuda",
            speaker_map: Optional[Dict[int, str]] = None,
            **kwargs
    ):
        """
        Initialize the transcriber chain.

        :param model: The model identifier to use for transcription
        :param use_gpu: Whether to utilize GPU acceleration
        :param device_type: The type of device to use, either "cuda" "cpu" or "mps"
        :param speaker_map: Mapping of channel numbers to speaker labels
        :param kwargs: Additional arguments passed to the parent ChainRunner
        :raises ValueError: If an invalid device_type is provided
        """
        super().__init__(**kwargs)

        if device_type not in ("cuda", "cpu", "mps"):
            raise ValueError(f"Invalid device_type: {device_type}. Must be 'cuda', 'cpu', or 'mps'.")

        self.transcriber = TranscriberPerChannel(
            n_channels=N_CHANNELS,
            api=True,
            # Uncomment and modify as needed:
            # transcriber_kwargs={
            #     "model": model,
            #     "use_gpu": use_gpu,
            #     "device_type": device_type,
            # }
        )

        self.channel_to_speaker_map = speaker_map or self.DEFAULT_SPEAKER_MAP
        logger.info(f"Initialized {self.__class__.__name__} with model: {model}")

    def _run(self, event) -> Dict:
        """
        Run the transcriber step.

        :param event: The event containing the transcription request
        :return: Dict containing the answer, sources, and extra data
        :raises RuntimeError: If transcription fails
        """
        audio_request = event.results.get("transcription_request")

        if not audio_request:
            return {"answer": "", "sources": ""}

        try:
            transcription_output = self.transcriber(audio_request) or {}
            output = {
                "answer": "",
                "sources": "",
                "extra_data": transcription_output
            }
            return output

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to transcribe audio: {str(e)}") from e

    def __repr__(self) -> str:
        """
        Return a string representation of the Transcribe instance.

        :return: String representation of the instance
        """
        return f"{self.__class__.__name__}(channels={N_CHANNELS})"