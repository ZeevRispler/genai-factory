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

from langchain_core.prompts.prompt import PromptTemplate

from genai_factory.chains.base import ChainRunner
from genai_factory.config import get_llm
from genai_factory.schemas import WorkflowEvent
from genai_factory.utils import logger

_prompt_template = """
Given the following conversation and a follow up request, generate a response that is relevant to the conversation.
Chat History:
{chat_history}
request: {question}
"""


class GenralLLM(ChainRunner):
    def __init__(self, llm=None, prompt_template=None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.prompt_template = prompt_template
        self._chain = None

    def post_init(self, mode="sync"):
        self.llm = self.llm or get_llm(self.context._config)
        classifier_prompt = PromptTemplate.from_template(
            self.prompt_template or _classifier_prompt_template
        )
        self._chain = classifier_prompt | self.llm

    def _run(self, event: WorkflowEvent):
        chat_history = str(event.conversation)
        logger.debug(f"Question: {event.query}\nChat history: {chat_history}")
        resp = self._chain.invoke(
            {
                "question": event.query,
                "chat_history": chat_history,
                "classifier_classes": self.classifier_classes,
            }
        )
        logger.debug(f"classifierd question: {resp}")
        return {"answer": resp}
