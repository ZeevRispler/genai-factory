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

from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple

from genai_factory.chains.base import ChainRunner
from audio_tools import VADBufferPerChannel, N_CHANNELS


@dataclass
class AudioEvent:
    """Data class representing an audio processing event."""
    audio_request: Optional[Any]
    results: Dict[str, Any]


@dataclass
class VADOutput:
    """Data class representing the output of VAD processing."""
    answer: str
    sources: str
    session_extra_data: Dict[str, Any]
    transcription_request: Any


class Vad(ChainRunner):
    """
    Voice Activity Detection processor that handles audio transcription.

    This class implements a Voice Activity Detection system that processes
    audio input and prepares it for transcription. It maintains a buffer
    per audio channel and processes incoming audio requests.
    """

    def __init__(
            self,
            buffer_size_seconds: float = 8.0,
            **kwargs
    ) -> None:
        """
        Initialize the VAD processor.

        Args:
            buffer_size_seconds (float): Maximum buffer size in seconds. Defaults to 8.0.
            **kwargs: Additional arguments passed to the parent ChainRunner.
        """
        super().__init__(**kwargs)
        self.vad_buffer = self._initialize_vad_buffer(buffer_size_seconds)

    def _initialize_vad_buffer(self, buffer_size_seconds: float) -> VADBufferPerChannel:
        """
        Initialize the VAD buffer with specified parameters.

        Args:
            buffer_size_seconds (float): Maximum buffer size in seconds.

        Returns:
            VADBufferPerChannel: Initialized VAD buffer object.
        """
        return VADBufferPerChannel(
            n_channels=N_CHANNELS,
            max_buffer_size_s=buffer_size_seconds,
        )

    def _process_audio_request(
            self,
            audio_request: Any,
            existing_buffer: Optional[Any]
    ) -> Tuple[Any, Any]:
        """
        Process the audio request using the VAD buffer.

        Args:
            audio_request: The audio data to process.
            existing_buffer: Any existing buffer data from previous processing.

        Returns:
            Tuple containing the transcription request and updated buffer.
        """
        return self.vad_buffer(
            request=audio_request,
            buffers=existing_buffer
        )

    def _create_empty_response(self) -> VADOutput:
        """
        Create an empty response for when no audio request is present.

        Returns:
            VADOutput: Empty response object.
        """
        return VADOutput(
            answer="",
            sources="",
            session_extra_data={"buffer": None},
            transcription_request=None
        )

    def _run(self, event: AudioEvent) -> Dict[str, Any]:
        """
        Process an audio event through the VAD system.

        This method handles the main processing logic for audio events,
        including buffer management and transcription request preparation.

        Args:
            event (AudioEvent): The audio event to process.

        Returns:
            Dict[str, Any]: Processing results including transcription request
                           and buffer state.
        """
        if not event.audio_request:
            return self._create_empty_response().__dict__

        existing_buffer = event.results.get("session_extra_data", {}).get("buffer")

        transcription_request, buffer = self._process_audio_request(
            event.audio_request,
            existing_buffer
        )

        return VADOutput(
            answer="filler text",  # TODO: Implement actual transcription
            sources="",
            session_extra_data={"buffer": buffer},
            transcription_request=transcription_request
        ).__dict__