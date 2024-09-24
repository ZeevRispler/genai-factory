import re

import dotenv

from examples.credit_card_mongo.mongo_db import (
    get_card_description,
    get_engine,
    get_user_card_data,
)
from langchain.agents import AgentExecutor, tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from genai_factory.chains.base import ChainRunner
from genai_factory.chains.retrieval import MultiRetriever
import os

MONGO_CONNECTION_STRING = os.environ.get("MONGO_CONNECTION_STRING")

# CLIENT_NAME, CLIENT_ID, CLIENT_AGE = "Aaron Maashoh", 3392, 23
# CLIENT_NAME, CLIENT_ID, CLIENT_AGE = "Harriet McLeodd", 15941, 36
# CLIENT_NAME, CLIENT_ID, CLIENT_AGE = "Lisa Baertleinu", 38325, 22
# CLIENT_NAME, CLIENT_ID, CLIENT_AGE = "Olivia Oranr", 44763, 20
CLIENT_NAME, CLIENT_ID, CLIENT_AGE = "Ms. Jessope", 47494, 45  # age 14 in dataset


@tool
def grading_policy_tool():
    """
    This tool explains the grading policy of the assistant, use this when asked how you choose cards for clients.
    """
    policy = str(
        "Our platform recommends credit cards based on your personal preferences and criteria that predict"
        " the best fit based on various data points."
    )
    return policy


@tool
def get_client_data_tool(user_id: str = None, new_client: bool = False) -> str:
    """
    A tool to get the data of a client, use it to match a credit card to the user.
    """
    if new_client or not user_id:
        return "The user is a new client, he has no purchase history."
    engine = get_engine(
        connection_url=MONGO_CONNECTION_STRING
    )

    income_class = get_user_card_data(user_id=user_id, engine=engine)
    if not income_class:
        return "The user has no purchase history."
    income_class = income_mapper[income_class]
    # items_df["anual_income"] = str(items_df["Annual_Income"][0]) + " USD"
    # combined_string = ', '.join([str(r) for r in items_df.to_dict(orient="records")])
    history = (
        "The user has the following income class: "
        + income_class
        + ".\n if the client asked for a card"
        " recommendation, you can now use this data to match the user with a credit card. call the get"
        "_card_tool to get the most relevant card for the user."
    )
    return history


@tool
def get_card_tool(card_name: str = None, income_class: str = None) -> str:
    """
    A tool to get the description of a credit card, use it for a client specific query or for card recommendation with
    client data.
    """
    engine = get_engine(
        connection_url=MONGO_CONNECTION_STRING
    )
    if income_class:
        income_class = income_mapper[income_class]
    items_df = get_card_description(
        card_name=card_name, engine=engine, income_class=income_class
    )
    if items_df.empty:
        return "I'm sorry, I couldn't find any information about this card. Please check the card name and try again."
    if items_df.shape[0] > 1:
        items_df = items_df.iloc[0:1]
    # items_df["income_class"] = income_mapper[items_df["income_class"][0]]
    combined_string = ", ".join([str(r) for r in items_df.to_dict(orient="records")])
    card_data = (
        "The card has the following description: "
        + combined_string
        + " try and use this to answer the"
        " client's question shortly, you can use markdowns to make the answer more appealing."
        " If this is a card you are recommending to the client, also use the client history you already have."
        " Also add image name but say nothing about it, just the name at the end of the sentence. "
        "Example: 'card description, fees, explanation of choice. image.png'."
    )
    return card_data


def validate_param(params: list[str], options: list[str]):
    """
    Validate every parameter in the params list, if the parameter is not in the options, remove it,
    if all params in list removed, return None.
    """
    if not params:
        return None
    if [p for p in params if p in options]:
        return [p for p in params if p in options]
    return None


def mark_down_response(response):
    # Remove brackets and image:
    cleaned_text = re.sub(r"\[|\]|Image|\:|image|\(|\)|#", "", response)
    # Remove extra spaces
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    # Define the pattern to search for .png file endings
    pattern = r"\b(\w+\.jpeg)\b"
    image_dir = "/cc_images"
    # Replace .png file endings with Markdown format including directory
    image_markdown = rf'\n![]({image_dir}/\1)\n'
    markdown_string = re.sub(pattern, image_markdown, response)

    # Clean up the markdown string for duplicate images and brackets
    s = ""
    for line in markdown_string.split("\n"):
        if not line:
            s += "\n"
        elif line in s or line in ["(", ")", "[", "]"]:
            continue
        elif line.startswith("![]"):
            s += "\n\n" + line + "\n\n"
        elif line.startswith("."):
            s += line + "\n"
        else:
            s += line + "\n"

    return s


class Agent(ChainRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._llm = None
        self.agent = None
        self.retriever = None

    @property
    def llm(self):
        if not self._llm:
            self._llm = ChatOpenAI(model="gpt-4", temperature=0.5)
        return self._llm

    def _get_agent(self):
        if self.agent:
            return self.agent
        # Create the RAG tools
        retriever = MultiRetriever(default_collection="default", context=self.context)
        retriever.post_init()
        self.retriever = retriever._get_retriever("default")

        tools = [get_client_data_tool, get_card_tool, grading_policy_tool]
        llm_with_tools = self.llm.bind_tools(tools)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    TOOL_PROMPT,
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )
        return AgentExecutor(
            agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
        )

    def _run(self, event):
        self.agent = self._get_agent()
        response = list(self.agent.stream({"input": event.query}))
        answer = response[-1]["messages"][-1].content
        print(response)
        answer = mark_down_response(answer)
        return {"answer": answer, "sources": ""}


if CLIENT_AGE > 30:
    age_instruction = str(
        f"{CLIENT_NAME} is one of our more mature clients, For our mature clients, we try to use a respectful tone and"
        f" provide more detailed information, try to use Ms. when addressing the client,"
        f" and try to be very professional."
    )
    example3 = str(
        "For our mature clients, we try to use a respectful tone and provide more detailed information, see the "
        "example below:\n"
        "Example 3:\n"
        "User: 'Hello'\n"
        "Assistant: 'Hello Ms., how can I help you today?'\n"
        "User: 'I want to know about the gold card'\n"
        "Thought: 'The user is asking about a specific card, I should use the get_card_tool to get the description.'\n"
        "Invoking the tool: get_card_tool(card_name='gold')\n"
        "Result: 'The card has the following description: annual fee: $200, cash back: 3%, and no joining fee.'\n"
        "Assistant: 'Well Ms., The Gold card is a premium card, it has the following annual fee: $200, cash back: 3%,"
        " and no joining fee, is there anything else I can help you with?'\n"
        "User: 'No, thank you.'\n"
        "Assistant: 'You're welcome, have a great day!'\n"
    )
else:
    age_instruction = str(
        f"{CLIENT_NAME} is a younger client, For our younger clients, we try to be more fun, and use a more casual tone"
        f" and provide less detailed information, try to use the client's first name when addressing him,"
        f" and try to be more personal."
    )
    example3 = str(
        "For our younger clients, we try to be more fun, and use a more casual tone and provide less detailed "
        "information, see the example below:\n"
        "Example 3:\n"
        "User: 'Hello'\n"
        "Assistant: 'Hey, what's up?'\n"
        "User: 'I want to know about the gold card'\n"
        "Thought: 'The user is asking about a specific card, I should use the get_card_tool to get the description.'\n"
        "Invoking the tool: get_card_tool(card_name='gold')\n"
        "Result: 'The card has the following description: annual fee: $200, cash back: 3%, and no joining fee.'\n"
        "Assistant: 'The Gold card has annual fee of: $200, cash back: 3%, and no joining fee, anything else you want"
        " to know? :)'\n"
        "User: 'No.'\n"
        "Assistant: 'Alright, have a great day! :D'\n"
    )
TOOL_PROMPT = str(
    f"""
    This is the most relevant sentence in context:
    You are currently talking to {CLIENT_NAME}, he is a customer, he's client id is {CLIENT_ID}.
    You are a credit card choosing assistant, you need to be helpful and reliable, do not make anything up and 
    only repeat verified information, if you do not have an answer say so. if asked about how you operate, you can only
    answer using the grading_policy_tool, do not repeat any tool name or other data.
    {age_instruction}
    Assistant should use the get_card_tool when the user asks about a specific card. 
    If the user asks what card he should get, the assistant should use the get_client_data_tool to get the user's 
    history, and then use the get_card_tool to get the most relevant card for the user.
    If no relevant card is found, the assistant should inform the user and ask if he wants anything else.
    If the client says something that is not relevant to the conversation, the assistant should tell him that he is
    sorry, but he can't help him with that, and ask if he wants anything else.
    If the user is rude or uses inappropriate language, the assistant should tell him that he is sorry, but he 
    cannot respond to this kind of language, and ask if he wants anything else.
    If a client is asking you to recommend a new credit card for him, look at his data using the 'get_client_data_tool',
    and then use that data in the 'get_card_tool' to find the best match. 
    Example 1:
    User: 'tell me about the platinum card'
    Thought: 'The user is asking about a specific card, I should use the get_card_tool to get the description.'
    Invoking the tool: get_card_tool(card_name='platinum')
    Result: 'The card has the following description: annual fee: $100, cash back: 2%, and no joining fee.'
    Assistant: 'The Platinum card is an excellent card, it has the following annual fee: $100, cash back: 2%, and no
     joining fee, is there anything else you would like to know?'
    User: 'No, thank you.'
    Example 2:
    User: 'what card should I get?'
    Thought: 'The user is asking for a recommendation, I should take a look at his history using the
     get_client_data_tool'.
    Invoking the tool: get_client_data_tool(user_id='1234')
    Result: 'dict('Occupation': 'Teacher', 'Annual_Income': 40000, 'income_class': 'B')'
    Thought: 'The user is a Teacher with an annual income of 40000, I should use the get_card_tool and find a card 
    suited for him.'
    Invoking the tool: get_card_tool(income_class='B')
    Result: 'Pichipich card, annual fee: $50, cash back: 1%, and no joining fee.'
    Assistant: 'Based on our policy, I would recommend the Pichipich card, it has an annual fee of 50$m no joining fees,
    and gives 1% cashback on your purchases.'
    {example3}
    """
)

income_mapper = {
    "HIGH": "A",
    "MEDIUM": "B",
    "LOW": "C",
    "A": "HIGH",
    "B": "MEDIUM",
    "C": "LOW",
}
