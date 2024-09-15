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

from mlrun.genai.api import router
from mlrun.genai.chains.base import HistorySaver, SessionLoader
from mlrun.genai.chains.refine import RefineQuery
from mlrun.genai.chains.retrieval import MultiRetriever
from mlrun.genai.pipelines import app_server

pipe_graph = [
    SessionLoader(),
    RefineQuery(),
    MultiRetriever(),
    HistorySaver(),
]


app_server.add_pipeline("default", pipe_graph)
app = app_server.to_fastapi(router=router)
