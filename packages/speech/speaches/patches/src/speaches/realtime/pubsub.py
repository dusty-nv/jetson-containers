from asyncio import Queue
from collections.abc import AsyncGenerator
import json
import logging
from pathlib import Path

from pydantic import BaseModel

from speaches.types.realtime import CLIENT_EVENT_TYPES, SERVER_EVENT_TYPES, Event

logger = logging.getLogger(__name__)

from typing import TypeVar, Generic, List, Set
T_PubSub = TypeVar("T_PubSub", bound="BaseModel")

class PubSub(Generic[T_PubSub]):
    def __init__(self) -> None:
        self.subscribers: Set[Queue[T_PubSub]] = set()
        self.events: List[T_PubSub] = []  # to store all events

    async def publish(self, event: T_PubSub) -> None:
        self.events.append(event)
        for subscriber in self.subscribers:
            await subscriber.put(event)

    def publish_nowait(self, event: T_PubSub) -> None:
        self.events.append(event)
        for subscriber in self.subscribers:
            subscriber.put_nowait(event)

    def subscribe(self) -> Queue[T_PubSub]:
        subscriber = Queue[T_PubSub]()
        self.subscribers.add(subscriber)
        return subscriber

    async def poll(self) -> AsyncGenerator[T_PubSub, None]:
        subscriber = Queue[T_PubSub]()
        self.subscribers.add(subscriber)
        try:
            while True:
                event = await subscriber.get()
                yield event.model_copy()
        finally:
            self.subscribers.remove(subscriber)
            logger.info("Subscriber removed")


class EventPubSub(PubSub[Event]):
    async def subscribe_to(self, event_type: str) -> AsyncGenerator[Event, None]:
        if event_type not in SERVER_EVENT_TYPES | CLIENT_EVENT_TYPES:
            raise ValueError(f"Invalid event type: {event_type}")
        subscriber = Queue[Event]()
        self.subscribers.add(subscriber)
        try:
            while True:
                event = await subscriber.get()
                if event.type == event_type:  # Only yield events matching the requested type
                    yield event.model_copy()
        finally:
            self.subscribers.remove(subscriber)
            logger.info(f"Subscriber for event type {event_type} removed")

    def dump_to_file(self, file_path: Path) -> None:
        with file_path.open("w") as f:
            f.write(json.dumps([event.model_dump() for event in self.events], indent=2))


# TODO: log delay between when message is added and the subscriber is notified
