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
from typing import List, Union

import mlrun.serving as mlrun_serving
from mlrun.utils import get_caller_globals

from genai_factory.config import WorkflowServerConfig
from genai_factory.controller_client import ControllerClient
from genai_factory.schemas import APIDictResponse, WorkflowType
from genai_factory.schemas import Workflow as WorkflowSchema
from genai_factory.sessions import SessionStore
import genai_factory

class Workflow:
    def __init__(
        self,
        name: str,
        version: str,
        workflow_type: WorkflowType,
        skeleton: Union[List[Union[mlrun_serving.states.FlowStep, dict]], dict],
        session_store: SessionStore,
        config: WorkflowServerConfig,
        client: ControllerClient,
        description: str = "",
        labels: dict = None,
        deployment: str = None,
    ):
        # Validate the skeleton:
        if not skeleton:
            raise ValueError("The workflow skeleton must not be empty")

        # Store parameters:
        self._name = name
        self._version = version
        self._workflow_type = workflow_type
        self._skeleton = skeleton
        self._session_store = session_store
        self._config = config
        self._labels = labels
        self._description = description
        self._deployment = deployment
        self._client = client

        # Prepare future instances:
        self._graph = None
        self._server = None

    def to_schema(self) -> WorkflowSchema:
        return WorkflowSchema(
            owner_id=self._client.owner_id,
            project_id=self._client.project_id,
            name=self._name,
            version=self._version,
            workflow_type=self._workflow_type,
            configuration=self.get_config(),
            graph=self._graph.to_dict(),
            labels=self._labels,
            description=self._description,
            deployment=self._deployment,
        )

    def set_deployment(self):
        self._deployment = os.path.join(
            self._config.deployment_url, f"api/workflows/{self._name}/infer"
        )

    def get_config(self):
        return self._config.workflows_kwargs.get(self._name, {})

    def build(self, config: WorkflowServerConfig, session_store: SessionStore):
        self._config = config
        self._session_store = session_store
        steps_config = self._config.workflows_kwargs.get(self._name, {}).get(
            "steps", {}
        )
        if isinstance(self._skeleton, list):
            self._graph = mlrun_serving.states.RootFlowStep()
            last_step = self.rec_build(last_step=self._graph, steps=self._skeleton, steps_config=steps_config)
            last_step.respond()
            return

        # Skeleton is a graph dictionary:
        self._graph = mlrun_serving.states.RootFlowStep.from_dict(self._skeleton)
        for step in self._graph:
            if step.name in getattr(steps_config, "steps", {}):
                step.class_args = steps_config["steps"][step.name]

    def rec_build(self, last_step, steps, steps_config):
        """
        Builds a workflow by connecting steps according to their order in the list.
        Handles sequential flows, parallel branches with tuples, and nested structures.

        Examples:
            rec_build(root, [step1, step2, step3], config)
            is equivalent to:
            root.to(step1)
            step1.to(step2)
            step2.to(step3)

            rec_build(root, [step1, (step2a, step2b), step3], config)
            is equivalent to:
            root.to(step1)
            step1.to(step2a)
            step1.to(step2b)
            step2a.to(step3)
            step2b.to(step3)

            rec_build(root, [(step1a, step1b), (step2a, step2b), step3], config)
            is equivalent to:
            root.to(step1a)
            root.to(step1b)
            step1a.to(step2a)
            step1a.to(step2b)
            step1b.to(step2a)
            step1b.to(step2b)
            step2a.to(step3)
            step2b.to(step3)

        :param last_step: The previous step to connect from (starting with RootFlowStep)
        :param steps: List of steps, can contain tuples for parallel execution
        :param steps_config: Configuration for steps including class args and parameters
        :returns: The last initialized step in the sequence
        """
        if not steps:
            raise ValueError("Cannot build an empty flow")

        def process_parallel_tuple(tuple_steps, prev_step, next_step=None):
            """Handles parallel execution paths by connecting steps in a tuple.

            Examples:
                process_parallel_tuple((step1, step2), prev, next)
                is equivalent to:
                prev.to(step1)
                prev.to(step2)
                step1.to(next)
                step2.to(next)

            :param tuple_steps: Tuple of steps to execute in parallel
            :param prev_step: Step that connects to all tuple steps
            :param next_step: Step that all tuple steps connect to (optional)
            :returns: The initialized next_step if provided, else last parallel step
            """
            initialized_parallel_steps = []

            # Connect previous step to each parallel step
            for step in tuple_steps:
                if prev_step != step:
                    initialized_step = self.connect_steps(cur_step=prev_step, next_step=step, steps_config=steps_config)
                    initialized_parallel_steps.append(initialized_step)
                else:
                    initialized_parallel_steps.append(step)

            # If there's a next step, connect all parallel steps to it
            if next_step:
                # Initialize next_step with first connection
                if initialized_parallel_steps[0] != next_step:
                    next_step = self.connect_steps(
                        cur_step=initialized_parallel_steps[0],
                        next_step=next_step,
                        steps_config=steps_config
                    )

                # Connect remaining parallel steps
                for step in initialized_parallel_steps[1:]:
                    if next_step != step:
                        self.connect_steps(cur_step=step, next_step=next_step, steps_config=steps_config)

                return next_step

            return initialized_parallel_steps[-1]

        cur_step = last_step

        for i in range(len(steps)):
            next_step = steps[i]

            if isinstance(next_step, tuple):
                # Handle parallel execution paths
                if i + 1 < len(steps):
                    cur_step = process_parallel_tuple(
                        tuple_steps=next_step,
                        prev_step=cur_step,
                        next_step=steps[i + 1]
                    )
                    i += 1  # Skip next step as it's been handled
                else:
                    # Handle parallel steps at the end
                    initialized_steps = []
                    for step in next_step:
                        if cur_step != step:
                            initialized_step = self.connect_steps(
                                cur_step=cur_step,
                                next_step=step,
                                steps_config=steps_config
                            )
                            initialized_steps.append(initialized_step)
                    cur_step = initialized_steps[-1]
            else:
                # Handle sequential steps
                if cur_step != next_step:
                    cur_step = self.connect_steps(
                        cur_step=cur_step,
                        next_step=next_step,
                        steps_config=steps_config
                    )

        return cur_step

    def connect_steps(self, cur_step, next_step, steps_config):
        """Connect two steps and return the initialized next step"""
        if isinstance(next_step, dict):
            step_name = next_step.get("name", next_step["class_name"])
            # Check if the current step is the same as the next step, to avoid loops
            if hasattr(cur_step, "class_name") and cur_step.to_dict()["class_name"].split('.')[-1] == step_name:
                return cur_step
            if step_name in getattr(steps_config, "steps", {}):
                next_step.update(steps_config["steps"][step_name])
            return cur_step.to(**next_step)
        else:
            step_name = next_step.name if hasattr(next_step, 'name') else str(next_step)
            # Check if the current step is the same as the next step, to avoid loops
            if hasattr(cur_step, "class_name") and cur_step.to_dict()["class_name"].split('.')[-1] == step_name:
                return cur_step
            if step_name in getattr(steps_config, "steps", {}):
                next_step.class_args = steps_config["steps"][step_name]
            return cur_step.to(next_step)

    @property
    def server(self) -> mlrun_serving.GraphServer:
        if self._server is None:
            namespace = get_caller_globals()
            server = mlrun_serving.create_graph_server(
                graph=self._graph,
                parameters={},
                verbose=self._config.verbose,
                graph_initializer=self.graph_initializer,
            )
            server.init_states(context=None, namespace=namespace)
            server.init_object(namespace)
            self._server = server
            return server
        return self._server

    def graph_initializer(self, server: mlrun_serving.GraphServer):
        context = server.context

        def register_prompt(
            name, template, description: str = None, llm_args: dict = None
        ):
            if not hasattr(context, "prompts"):
                context.prompts = {}
            context.prompts[name] = (template, llm_args)

        if getattr(context, "_config", None) is None:
            context._config = self._config
        if getattr(context, "session_store", None) is None:
            context.session_store = self._session_store

    def run(self, event, db_session=None):
        # todo: pass sql db_session to steps via context or event
        server = self.server
        try:
            resp = server.test("", body=event)
        except Exception as e:
            server.wait_for_completion()
            raise e
        return APIDictResponse(
            success=True,
            data={
                "answer": resp.results["answer"],
                "sources": resp.results["sources"],
                "returned_state": {},
            },
        )
