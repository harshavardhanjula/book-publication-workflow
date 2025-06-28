"""Utility functions for the scraper module."""
import asyncio
import functools
import logging
from pathlib import Path
from typing import Any, Callable, Coroutine, TypeVar, cast

from loguru import logger

T = TypeVar('T')


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging with loguru.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger.remove()  # Remove default handler
    logger.add(
        "scraper.log",
        rotation="10 MB",
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )


def ensure_directory_exists(path: str) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory
        
    Returns:
        Path object for the directory
    """
    path_obj = Path(path).resolve()
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def async_retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying async functions on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (attempt + 1)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {str(e)}. "
                            f"Retrying in {wait_time:.1f} seconds..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
                        raise last_exception
            raise last_exception  # This line should never be reached
        return cast(Callable[..., Coroutine[Any, Any, T]], wrapper)
    return decorator


async def run_async(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a synchronous function in an executor.
    
    Args:
        func: Synchronous function to run
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function call
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, functools.partial(func, *args, **kwargs)
    )
