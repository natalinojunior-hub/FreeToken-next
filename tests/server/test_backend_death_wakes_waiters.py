import asyncio

import pytest

from freetoken.server.api_server import FrontendManager


def test_backend_death_fails_inflight_generation():
    """A dead backend sends no more acks: waiters must fail instead of hanging (and holding
    uvicorn's graceful stop open forever)."""

    async def main():
        state = FrontendManager(config=None, send_tokenizer=None, recv_tokenizer=None)
        state._loop = asyncio.get_running_loop()
        state.ack_map[7], state.event_map[7] = [], asyncio.Event()
        waiter = asyncio.ensure_future(state.wait_for_ack(7).__anext__())
        await asyncio.sleep(0)
        state.fatal_error = "scheduler exited (exitcode=-6)"
        state.wake_generation_waiters()
        with pytest.raises(RuntimeError, match="backend died"):
            await asyncio.wait_for(waiter, 2)

    asyncio.run(main())
