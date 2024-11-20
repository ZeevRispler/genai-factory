from pydoc_data.topics import topics
import json
from storey.flow import (
    Context,
    ParallelExecution,
    ParallelExecutionRunnable,
    ReifyMetadata,
    Rename,
    _ConcurrentJobExecution,
)
from langchain_core.language_models.llms import LLM
from langchain_core.prompts.prompt import PromptTemplate

from genai_factory.chains.base import ChainRunner
from genai_factory.config import get_llm, WorkflowServerConfig
from genai_factory.schemas import WorkflowEvent
import time
class dataExtractor(ParallelExecutionRunnable):
    execution_mechanism = "threading"
    _prompt_template = str("You are a helpful AI, given the following conversation piece,"
                        " extract the relevant data from the follow up request, by specifyng 'yes' or 'no' for each of "
                        "the following topics (if mentioned):{topics}"
                        "Return the extracted data in a structured format."
                        "For example, if the follow up request is 'I am a 30 and smoke from time to time', you should"
                        " return: 'smoking': 'yes', 'age': '30'"
                        "Another example, if the follow up request is: 'Hello, my name is Tom', you should return: ''. "
                        "Input: {question} Extracted data:")
    def __init__(self,
                 llm = "gpt-4o-mini",
                 prompt_template: str = None,
                 **kwargs,
                 ):
        """
        :param llm:             The llm to use, if None a default llm will be used from config.
        :param prompt_template: A template for the prompt to use for the llm, if None a default prompt will be used.
        """
        super().__init__(**kwargs)
        self.llm = llm
        self.prompt_template = prompt_template
        self._chain = None
        self.topics = ["gender", "age", "smoking", "oncology screening", "cancer in family", "sale formalized"]
        classifier_prompt = PromptTemplate.from_template(
            self.prompt_template or self._prompt_template
        )
        llm_args = {
            "class_name": "langchain_openai.ChatOpenAI",
            "temperature": 0.3,
            "model_name": self.llm,
        }
        self._chain = classifier_prompt | get_llm(WorkflowServerConfig(), llm_args=llm_args)


    def run(self, event: WorkflowEvent):
        """
        Classify a given text, to of out of a list of possible classes.
        :param event: The event of the workflow, containing the conversation and the text to classify.
        :return:      An event, containing the answer and the sources.
        """
        resp = self._chain.invoke(
            {
                "question": event.body.query,
                "topics": str(self.topics),
            }
        )
        # Think if we need to store this, or just generate each time and let frontend handle it
        data_extracted = event.body.results.get("data_extracted", {}) or {topic: "NA" for topic in self.topics}
        try:
            resp = json.loads(resp.content)
            for topic in resp.keys():
                if data_extracted[topic] is not "yes":
                    data_extracted[topic] = resp[topic]

            print(f"Extracted data: {data_extracted}")
            return {"data_extracted": data_extracted, "sources": "", "answer": event.body.query}
        except Exception as e:
            print(f"Error extracting data: {e}")
            return {"data_extracted": data_extracted, "sources": "", "answer": event.body.query}

class sentimentAnalysis(ParallelExecutionRunnable):
    execution_mechanism = "threading"
    _prompt_template = str(
        "you are a helpful AI, given the following conversation piece,"
        " analyze it's sentiment to one of: positive, negative or natural.\n"
        "Return your answer in one word, input: {question} Sentiment:")
    def __init__(self,
                 llm = "gpt-4o-mini",
                 prompt_template: str = None,
                 **kwargs,
                 ):
        """
        :param llm:             The llm to use, if None a default llm will be used from config.
        :param prompt_template: A template for the prompt to use for the llm, if None a default prompt will be used.
        """
        super().__init__(**kwargs)
        self.llm = llm
        self.prompt_template = prompt_template
        classifier_prompt = PromptTemplate.from_template(
            self.prompt_template or self._prompt_template
        )
        llm_args = {
            "class_name": "langchain_openai.ChatOpenAI",
            "temperature": 0.3,
            "model_name": self.llm,
        }
        self._chain = classifier_prompt | get_llm(WorkflowServerConfig(), llm_args=llm_args)


    def run(self, event: WorkflowEvent):
        """
        Classify a given text, to of out of a list of possible classes.
        :param event: The event of the workflow, containing the conversation and the text to classify.
        :return:      An event, containing the answer and the sources.
        """
        resp = self._chain.invoke(
            {
                "question": event.body.query,
            }
        )
        print(f"classified question: {resp.content}")
        return {"sentiment": resp.content}

