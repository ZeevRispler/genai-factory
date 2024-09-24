from examples.credit_card_mongo.banking_agent import Agent
from genai_factory.chains.base import HistorySaver, SessionLoader
from genai_factory.chains.refine import RefineQuery
from genai_factory.workflows import workflow_server

workflow_graph = [
    SessionLoader(),
    RefineQuery(),
    Agent(),
    HistorySaver(),
]

workflow_server.add_workflow(
    name="default",
    graph=workflow_graph,
    workflow_type="application",
)
