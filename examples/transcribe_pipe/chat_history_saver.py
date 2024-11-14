
from genai_factory.chains.base import ChainRunner
from genai_factory.schemas import WorkflowEvent

DEFAULT_SPEAKER_MAP = {
    0: "Agent",
    1: "User"
}

class ChatHistorySaver(ChainRunner):
    def __init__(
        self,
        answer_key: str = None,
        question_key: str = None,
        save_sources: str = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.answer_key = answer_key
        self.question_key = question_key
        self.save_sources = save_sources

    async def _run(self, event: WorkflowEvent):
        # Get the transcription results from the previous step, stored in extra data
        messages = event.results.get("extra_data", {})

        # Sort the messages by timestamp
        sorted_list = [
            {**value, 'role': key}
            for key, value in messages.items()
        ]
        sorted_list.sort(key=lambda x: x['timestamp'])

        # Add the messages to the conversation, replacing the last message if it has the same timestamp as the last
        # message because it means it was a continuation of the same message
        for item in sorted_list:
            if item["text"] == "":
                continue
            if len(event.conversation) > 0:
                if DEFAULT_SPEAKER_MAP[item['role']] == event.conversation[-1].role:
                    if not str(item['timestamp']) == str(event.conversation[-1].extra_data["timestamp"]):
                        event.conversation[-1].content = item['text']
                        continue
            event.conversation.add_message(
                role=DEFAULT_SPEAKER_MAP[item['role']],
                content=item['text'],
                extra_data={"timestamp": item["timestamp"]})

        self.context.session_store.save(event)
        print("-"*80)
        print(event.conversation)
        print("-"*80)

        return event.results