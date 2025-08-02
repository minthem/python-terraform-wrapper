import asyncio
import weakref
from contextlib import asynccontextmanager


class AsyncResourceLockManager:
    """
    A lock manager that provides per-resource asyncio locks for safe concurrent access.

    Locks are stored in a WeakValueDictionary, allowing automatic garbage collection
    when no longer referenced. Each resource ID is mapped to a single asyncio.Lock instance.

    This class is useful for coordinating asynchronous access to shared resources
    identified by string keys, ensuring exclusive usage per resource.
    """

    def __init__(self):
        """Initialize the lock manager with a weak reference dictionary to hold locks."""
        self._locks = weakref.WeakValueDictionary()

    def get_lock(self, resource_id: str) -> asyncio.Lock:
        """
        Retrieve or create an asyncio.Lock for the given resource ID.

        Lock creation is not thread-safe, but in typical asyncio single-threaded execution,
        race conditions are unlikely.

        Args:
            resource_id: A string identifier for the resource to lock.

        Returns:
            An asyncio.Lock associated with the resource.
        """
        lock = self._locks.get(resource_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[resource_id] = lock
        return lock

    async def acquire(self, resource_id: str) -> asyncio.Lock:
        """
        Acquire the lock for the given resource ID.

        This method blocks until the lock is available.

        Args:
            resource_id: A string identifier for the resource to lock.

        Returns:
            The acquired asyncio.Lock.
        """
        lock = self.get_lock(resource_id)
        await lock.acquire()
        return lock

    async def release(self, resource_id: str):
        """
        Release the lock for the given resource ID, if it is currently held.

        Args:
            resource_id: A string identifier for the resource to unlock.
        """
        lock = self._locks.get(resource_id)
        if lock and lock.locked():
            lock.release()

    @asynccontextmanager
    async def context(self, resource_id: str):
        """
        Async context manager to acquire and release the lock for a resource ID.

        Ensures that the lock is released even if exceptions occur within the context.

        Args:
            resource_id: A string identifier for the resource to lock.

        Yields:
            None. Code inside the 'async with' block runs while the lock is held.
        """
        lock = self.get_lock(resource_id)
        await lock.acquire()
        try:
            yield
        finally:
            lock.release()
