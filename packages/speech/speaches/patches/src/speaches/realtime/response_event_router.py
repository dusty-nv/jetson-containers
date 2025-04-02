from __future__ import annotations

import asyncio
from contextlib import contextmanager
import logging
from typing import TYPE_CHECKING
from typing import TypeVar, Generator

T_OutputItem = TypeVar("T_OutputItem", bound="ServerConversationItem")
T_Content = TypeVar("T_Content", bound="ConversationItemContentText | ConversationItemContentAudio")

import openai
from openai.types.beta.realtime.error_event import Error
from pydantic import BaseModel

from speaches.realtime.chat_utils import (
    create_completion_params,
    items_to_chat_messages,
)
from speaches.realtime.event_router import EventRouter
from speaches.realtime.session_event_router import unsupported_field_error, update_dict
from speaches.realtime.utils import generate_response_id, task_done_callback
from speaches.types.realtime import (
    ConversationItemContentAudio,
    ConversationItemContentText,
    ConversationItemFunctionCall,
    ConversationItemMessage,
    ErrorEvent,
    RealtimeResponse,
    Response,
    # TODO: RealtimeResponseStatus,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ServerConversationItem,
    create_server_error,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from openai.resources.chat import AsyncCompletions
    from openai.types.chat import ChatCompletionChunk

    from speaches.realtime.context import SessionContext
    from speaches.realtime.conversation_event_router import Conversation
    from speaches.realtime.pubsub import EventPubSub

logger = logging.getLogger(__name__)

event_router = EventRouter()

# TODO: start using this error
conversation_already_has_active_response_error = Error(
    type="invalid_request_error",
    message="Conversation already has an active response",
)


class ChoiceDeltaAudio(BaseModel):
    id: str | None = None
    transcript: str | None = None
    data: str | None = None
    expires_at: int | None = None


class ResponseHandler:
    def __init__(
        self,
        *,
        completion_client: AsyncCompletions,
        model: str,
        configuration: Response,
        conversation: Conversation,
        pubsub: EventPubSub,
    ) -> None:
        self.id = generate_response_id()
        self.completion_client = completion_client
        self.model = model  # NOTE: unfortunatly `Response` doesn't have a `model` field
        self.configuration = configuration
        self.conversation = conversation
        self.pubsub = pubsub
        self.response = RealtimeResponse(
            id=self.id,
            status="incomplete",
            output=[],
            modalities=configuration.modalities,
        )
        self.task: asyncio.Task[None] | None = None

    @contextmanager
    def add_output_item(self, item: T_OutputItem) -> Generator[T_OutputItem, None, None]:
        self.response.output.append(item)
        self.pubsub.publish_nowait(ResponseOutputItemAddedEvent(response_id=self.id, item=item))
        yield item
        assert item.status == "incomplete", item
        item.status = "completed"
        self.pubsub.publish_nowait(ResponseOutputItemDoneEvent(response_id=self.id, item=item))
        self.response.status = "completed"
        self.pubsub.publish_nowait(ResponseDoneEvent(response=self.response))

    @contextmanager
    def add_item_content(
        self, item: ConversationItemMessage, content: T_Content
    ) -> Generator[T_Content, None, None]:
        item.content.append(content)
        self.pubsub.publish_nowait(
            ResponseContentPartAddedEvent(response_id=self.id, item_id=item.id, part=content.to_part())
        )
        yield content
        self.pubsub.publish_nowait(
            ResponseContentPartDoneEvent(response_id=self.id, item_id=item.id, part=content.to_part())
        )

    async def conversation_item_message_text_handler(self, chunk_stream: AsyncGenerator[ChatCompletionChunk]) -> None:
        with self.add_output_item(ConversationItemMessage(role="assistant", status="incomplete", content=[])) as item:
            self.conversation.create_item(item)

            with self.add_item_content(item, ConversationItemContentText(text="")) as content:
                async for chunk in chunk_stream:
                    assert len(chunk.choices) == 1, chunk
                    choice = chunk.choices[0]

                    if choice.delta.content is not None:
                        content.text += choice.delta.content
                        self.pubsub.publish_nowait(
                            ResponseTextDeltaEvent(item_id=item.id, response_id=self.id, delta=choice.delta.content)
                        )

                self.pubsub.publish_nowait(
                    ResponseTextDoneEvent(item_id=item.id, response_id=self.id, text=content.text)
                )

    async def conversation_item_message_audio_handler(self, chunk_stream: AsyncGenerator[ChatCompletionChunk]) -> None:
        with self.add_output_item(ConversationItemMessage(role="assistant", status="incomplete", content=[])) as item:
            self.conversation.create_item(item)

            with self.add_item_content(item, ConversationItemContentAudio(audio="", transcript="")) as content:
                async for chunk in chunk_stream:
                    assert len(chunk.choices) == 1, chunk
                    choice = chunk.choices[0]

                    audio = getattr(choice.delta, "audio", None)
                    if audio is None:
                        logger.warning(f"Could not extract audio from chunk: {chunk}")
                        continue
                    assert isinstance(audio, dict), chunk
                    audio = ChoiceDeltaAudio(**audio)
                    if audio.transcript is not None:
                        self.pubsub.publish_nowait(
                            ResponseAudioTranscriptDeltaEvent(
                                item_id=item.id, response_id=self.id, delta=audio.transcript
                            )
                        )
                        content.transcript += audio.transcript

                    if audio.data is not None:
                        self.pubsub.publish_nowait(
                            ResponseAudioDeltaEvent(item_id=item.id, response_id=self.id, delta=audio.data)
                        )
                        # NOTE: we explicitly don't append the audio data to the `audio` field

                self.pubsub.publish_nowait(ResponseAudioDoneEvent(item_id=item.id, response_id=self.id))
                self.pubsub.publish_nowait(
                    ResponseAudioTranscriptDoneEvent(
                        item_id=item.id, response_id=self.id, transcript=content.transcript
                    )
                )

    async def conversation_item_function_call_handler(self, chunk_stream: AsyncGenerator[ChatCompletionChunk]) -> None:
        chunk = await anext(chunk_stream)

        assert len(chunk.choices) == 1, chunk
        choice = chunk.choices[0]
        assert choice.delta.tool_calls is not None and len(choice.delta.tool_calls) == 1, chunk
        tool_call = choice.delta.tool_calls[0]
        assert (
            tool_call.id is not None
            and tool_call.function is not None
            and tool_call.function.name is not None
            and tool_call.function.arguments is not None
        ), chunk
        item = ConversationItemFunctionCall(
            status="incomplete",
            call_id=tool_call.id,
            name=tool_call.function.name,
            arguments=tool_call.function.arguments,
        )
        assert item.call_id is not None and item.arguments is not None and item.name is not None, item

        with self.add_output_item(item):
            self.conversation.create_item(item)

            async for chunk in chunk_stream:
                assert len(chunk.choices) == 1, chunk
                choice = chunk.choices[0]

                if choice.delta.tool_calls is not None:
                    assert len(choice.delta.tool_calls) == 1, chunk
                    tool_call = choice.delta.tool_calls[0]
                    assert tool_call.function is not None and tool_call.function.arguments is not None, chunk
                    self.pubsub.publish_nowait(
                        ResponseFunctionCallArgumentsDeltaEvent(
                            item_id=item.id,
                            response_id=self.id,
                            call_id=item.call_id,
                            delta=tool_call.function.arguments,
                        )
                    )
                    item.arguments += tool_call.function.arguments

            self.pubsub.publish_nowait(
                ResponseFunctionCallArgumentsDoneEvent(
                    arguments=item.arguments, call_id=item.call_id, item_id=item.id, response_id=self.id
                )
            )

    async def generate_response(self) -> None:
        try:
            completion_params = create_completion_params(
                self.model,
                list(items_to_chat_messages(self.configuration.input)),
                self.configuration,
            )
            chunk_stream = await self.completion_client.create(**completion_params)
            chunk = await anext(chunk_stream)
            if chunk.choices[0].delta.tool_calls is not None:
                handler = self.conversation_item_function_call_handler
            elif self.configuration.modalities == ["text"]:
                handler = self.conversation_item_message_text_handler
            else:
                handler = self.conversation_item_message_audio_handler

            async def merge_chunks_and_chunk_stream(
                *chunks: ChatCompletionChunk, chunk_stream: openai.AsyncStream[ChatCompletionChunk]
            ) -> AsyncGenerator[ChatCompletionChunk]:
                for chunk in chunks:
                    yield chunk
                async for chunk in chunk_stream:
                    yield chunk

            await handler(merge_chunks_and_chunk_stream(chunk, chunk_stream=chunk_stream))
        except openai.APIError as e:
            logger.exception("Error while generating response")
            self.pubsub.publish_nowait(
                ErrorEvent(error=Error(type="server_error", message=f"{type(e).__name__}: {e.message}"))
            )
            raise

    def start(self) -> None:
        assert self.task is None
        self.task = asyncio.create_task(self.generate_response())
        self.task.add_done_callback(task_done_callback)

    def stop(self) -> None:
        assert self.task is not None
        self.task.cancel()


@event_router.register("response.create")
async def handle_response_create_event(ctx: SessionContext, event: ResponseCreateEvent) -> None:
    if ctx.response is not None:
        ctx.response.stop()

    configuration = Response(
        conversation="auto", input=list(ctx.conversation.items.values()), **ctx.session.model_dump()
    )
    if event.response is not None:
        if event.response.conversation is not None:
            ctx.pubsub.publish_nowait(unsupported_field_error("response.conversation"))
        if event.response.input is not None:
            ctx.pubsub.publish_nowait(unsupported_field_error("response.input"))
        if event.response.output_audio_format is not None:
            ctx.pubsub.publish_nowait(unsupported_field_error("response.output_audio_format"))
        if event.response.metadata is not None:
            ctx.pubsub.publish_nowait(unsupported_field_error("response.metadata"))

        configuration_dict = configuration.model_dump()
        configuration_update_dict = event.response.model_dump(
            exclude_none=True, exclude={"conversation", "input", "output_audio_format", "metadata"}
        )
        logger.debug(f"Applying response configuration update: {configuration_update_dict}")
        logger.debug(f"Response configuration before update: {configuration_dict}")
        updated_configuration = update_dict(configuration_dict, configuration_update_dict)
        logger.debug(f"Response configuration after update: {updated_configuration}")
        configuration = Response(**updated_configuration)

    ctx.response = ResponseHandler(
        completion_client=ctx.completion_client,
        model=ctx.session.model,
        configuration=configuration,
        conversation=ctx.conversation,
        pubsub=ctx.pubsub,
    )
    ctx.pubsub.publish_nowait(ResponseCreatedEvent(response=ctx.response.response))
    ctx.response.start()
    assert ctx.response.task is not None
    await ctx.response.task
    ctx.response = None


@event_router.register("response.cancel")
def handle_response_cancel_event(ctx: SessionContext, event: ResponseCancelEvent) -> None:
    ctx.pubsub.publish_nowait(
        create_server_error(f"Handling of the '{event.type}' event is not implemented.", event_id=event.event_id)
    )
