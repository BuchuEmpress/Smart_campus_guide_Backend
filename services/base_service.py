import asyncio
from typing import Any, Callable, Coroutine

class BaseService:
    """
    Base service class providing common utilities for all services.
    """
    
    async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run a synchronous function in a separate thread to avoid blocking the event loop.
        If the function is a coroutine, await it directly.
        """
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return await asyncio.to_thread(func, *args, **kwargs)
