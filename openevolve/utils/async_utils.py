"""
Async utilities for OpenEvolve
"""
import asyncio
import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


def run_in_executor(f: Callable) -> Callable:
    """
    Decorator to run a synchronous function in an executor
    
    Args:
        f: Function to decorate
        
    Returns:
        Decorated function that runs in an executor
    """
    @functools.wraps(f)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            functools.partial(f, *args, **kwargs)
        )
    return wrapper


async def gather_with_concurrency(
    n: int, 
    *tasks: asyncio.Future,
    return_exceptions: bool = False
) -> List[Any]:
    """
    Run tasks with a concurrency limit
    
    Args:
        n: Maximum number of tasks to run concurrently
        *tasks: Tasks to run
        return_exceptions: Whether to return exceptions instead of raising them
        
    Returns:
        List of task results
    """
    semaphore = asyncio.Semaphore(n)
    
    async def sem_task(task: asyncio.Future) -> Any:
        async with semaphore:
            return await task
    
    return await asyncio.gather(
        *(sem_task(task) for task in tasks),
        return_exceptions=return_exceptions
    )


async def retry_async(
    coro: Callable,
    *args: Any,
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Union[Exception, tuple] = Exception,
    **kwargs: Any
) -> Any:
    """
    Retry an async function with exponential backoff
    
    Args:
        coro: Coroutine function to retry
        *args: Arguments to pass to the coroutine
        retries: Maximum number of retries
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay between retries
        exceptions: Exception(s) to catch
        **kwargs: Keyword arguments to pass to the coroutine
        
    Returns:
        Result of the coroutine
        
    Raises:
        The last exception caught if all retries fail
    """
    last_exception = None
    current_delay = delay
    
    for i in range(retries + 1):
        try:
            return await coro(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if i < retries:
                logger.warning(
                    f"Retry {i+1}/{retries} failed with {type(e).__name__}: {str(e)}. "
                    f"Retrying in {current_delay:.2f}s..."
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(
                    f"All {retries+1} attempts failed. Last error: {type(e).__name__}: {str(e)}"
                )
    
    if last_exception:
        raise last_exception
    
    return None  # Should never reach here


class TaskPool:
    """
    A simple task pool for managing and limiting concurrent tasks
    """
    
    def __init__(self, max_concurrency: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.tasks: List[asyncio.Task] = []
    
    async def run(self, coro: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Run a coroutine in the pool
        
        Args:
            coro: Coroutine function to run
            *args: Arguments to pass to the coroutine
            **kwargs: Keyword arguments to pass to the coroutine
            
        Returns:
            Result of the coroutine
        """
        async with self.semaphore:
            return await coro(*args, **kwargs)
    
    def create_task(self, coro: Callable, *args: Any, **kwargs: Any) -> asyncio.Task:
        """
        Create and track a task in the pool
        
        Args:
            coro: Coroutine function to run
            *args: Arguments to pass to the coroutine
            **kwargs: Keyword arguments to pass to the coroutine
            
        Returns:
            Task object
        """
        task = asyncio.create_task(self.run(coro, *args, **kwargs))
        self.tasks.append(task)
        task.add_done_callback(lambda t: self.tasks.remove(t))
        return task
    
    async def wait_all(self) -> None:
        """Wait for all tasks in the pool to complete"""
        if self.tasks:
            await asyncio.gather(*self.tasks)
    
    async def cancel_all(self) -> None:
        """Cancel all tasks in the pool"""
        for task in self.tasks:
            task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
