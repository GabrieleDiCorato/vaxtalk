from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from google.adk.agents import parallel_agent as adk_parallel
from google.adk.events.event import Event
from google.adk.utils.context_utils import Aclosing


async def _merge_agent_run_with_aclosing(
    agent_runs: list[AsyncGenerator[Event, None]],
) -> AsyncGenerator[Event, None]:
    if not agent_runs:
        return

    sentinel = object()
    queue: asyncio.Queue[tuple[object, asyncio.Event | None]] = asyncio.Queue()

    async def process_an_agent(events_for_one_agent: AsyncGenerator[Event, None]):
        try:
            async with Aclosing(events_for_one_agent) as agen:
                async for event in agen:
                    resume_signal = asyncio.Event()
                    await queue.put((event, resume_signal))
                    await resume_signal.wait()
        except asyncio.CancelledError:
            raise
        except GeneratorExit:
            return
        except BaseException as exc:  # noqa: BLE001 - funnel errors to consumer
            await queue.put((exc, None))
        finally:
            await queue.put((sentinel, None))

    async with asyncio.TaskGroup() as task_group:
        workers: list[asyncio.Task[None]] = []
        for events_for_one_agent in agent_runs:
            workers.append(task_group.create_task(process_an_agent(events_for_one_agent)))

        sentinel_count = 0
        pending_error: BaseException | None = None
        closing_requested = False
        while sentinel_count < len(agent_runs) and not closing_requested:
            event, resume_signal = await queue.get()
            if event is sentinel:
                sentinel_count += 1
                continue
            if isinstance(event, BaseException):
                pending_error = event
                break

            try:
                yield event
            except GeneratorExit:
                closing_requested = True
            finally:
                if resume_signal:
                    resume_signal.set()

        if closing_requested:
            for worker in workers:
                worker.cancel()
            return

        if pending_error:
            raise pending_error


async def _merge_agent_run_pre_3_11_with_aclosing(
    agent_runs: list[AsyncGenerator[Event, None]],
) -> AsyncGenerator[Event, None]:
    if not agent_runs:
        return

    sentinel = object()
    queue: asyncio.Queue[tuple[object, asyncio.Event | None]] = asyncio.Queue()

    def propagate_exceptions(tasks: list[asyncio.Task[None]]) -> None:
        for task in tasks:
            if task.done():
                task.result()

    async def process_an_agent(events_for_one_agent: AsyncGenerator[Event, None]):
        try:
            async with Aclosing(events_for_one_agent) as agen:
                async for event in agen:
                    resume_signal = asyncio.Event()
                    await queue.put((event, resume_signal))
                    await resume_signal.wait()
        except asyncio.CancelledError:
            raise
        except GeneratorExit:
            return
        except BaseException as exc:  # noqa: BLE001 - pass failure upstream
            await queue.put((exc, None))
        finally:
            await queue.put((sentinel, None))

    tasks: list[asyncio.Task[None]] = []
    pending_error: BaseException | None = None
    try:
        for events_for_one_agent in agent_runs:
            tasks.append(asyncio.create_task(process_an_agent(events_for_one_agent)))

        sentinel_count = 0
        closing_requested = False
        while sentinel_count < len(agent_runs) and not closing_requested:
            propagate_exceptions(tasks)
            event, resume_signal = await queue.get()
            if event is sentinel:
                sentinel_count += 1
                continue
            if isinstance(event, BaseException):
                pending_error = event
                break

            try:
                yield event
            except GeneratorExit:
                closing_requested = True
            finally:
                if resume_signal:
                    resume_signal.set()
    finally:
        for task in tasks:
            task.cancel()

    if closing_requested:
        return

    if pending_error:
        raise pending_error


def patch_parallel_agent(logger=None) -> None:
    if getattr(adk_parallel, "_vaxtalk_parallel_patch", False):
        return

    adk_parallel._merge_agent_run = _merge_agent_run_with_aclosing
    adk_parallel._merge_agent_run_pre_3_11 = _merge_agent_run_pre_3_11_with_aclosing
    adk_parallel._vaxtalk_parallel_patch = True
    if logger:
        logger.info("Patched ParallelAgent merge loops to preserve tracing context")
