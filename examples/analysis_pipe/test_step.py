# Copyright 2023 Iguazio
#
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

from langchain_core.language_models.llms import LLM
from langchain_core.prompts.prompt import PromptTemplate

from genai_factory.chains.base import ChainRunner
from genai_factory.config import get_llm
from genai_factory.schemas import WorkflowEvent
from genai_factory.utils import logger


class TestStep(ChainRunner):

    def _run(self, event: WorkflowEvent) -> dict:
        """
        Generate a response to a given text based on the conversation history.
        :param event: The event of the workflow, containing the conversation and the query.
        :return:      An event, containing the answer and the sources.
        """
        sentiment = event.kwargs.get("outputs").get("sentimeter").get("sentiment")
        data_extracted = event.kwargs.get("outputs").get("data_collector").get("data_extracted")
        print(f"Sentiment: {sentiment}")
        print(f"Data Extracted: {data_extracted}")
        # return_dict = {"answer": "JUNK", **event.to_dict()}
        return_dict = {**event.kwargs.get("outputs").get("data_collector"), **event.kwargs.get("outputs").get("sentimeter")}
        return return_dict



