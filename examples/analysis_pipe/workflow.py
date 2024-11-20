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
import os
import sys
# Add the directory containing your modules to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


from genai_factory.chains.base import HistorySaver, SessionLoader
from genai_factory.chains.refine import RefineQuery
from genai_factory import workflow_server
# from .llm_invoke import GeneralLLMInvoke
from test_step import TestStep
from parallel_step import ParallelExecution, dataExtractor, sentimentAnalysis

runnables = [dataExtractor(name="data_collector"), sentimentAnalysis(name="sentimeter")]
translation_prompt = str("You are a translator. Please translate the following text from spanish to english"
                         " with as little changes to the meaning as possible:{question}."
                         "you can use this as context: {chat_history}")
workflow_graph = [
    SessionLoader(),
    # GeneralLLMInvoke(prompt_template=translation_prompt), #translate
    ParallelExecution(runnables=runnables),
    # GeneralLLMInvoke(), #recommender
    TestStep(),
    HistorySaver(),
]



workflow_server.add_workflow(
    name="default",
    graph=workflow_graph,
    workflow_type="application",
)