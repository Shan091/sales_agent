import hashlib
import logging
import uuid
from typing import Optional
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

# Lua script for owner-safe lock release (H6 fix)
# Only deletes the lock if the current value matches the owner's UUID
RELEASE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


class CacheService:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    async def acquire_lock(self, thread_id: str, timeout: int = 30) -> Optional[str]:
        """
        Attempts to acquire a distributed mutex lock for a given thread_id.
        Returns the lock_value (UUID) if acquired, None if already held.
        The caller MUST pass the returned lock_value to release_lock().
        """
        lock_key = f"wa_mutex:{thread_id}"
        lock_value = str(uuid.uuid4())
        acquired = await self.redis.set(lock_key, lock_value, nx=True, ex=timeout)
        if acquired:
            return lock_value
        return None

    async def release_lock(self, thread_id: str, lock_value: str):
        """
        Releases the mutex lock ONLY if the caller owns it (value matches).
        Prevents Worker A from accidentally deleting Worker B's lock after TTL expiry.
        """
        lock_key = f"wa_mutex:{thread_id}"
        await self.redis.eval(RELEASE_LOCK_SCRIPT, 1, lock_key, lock_value)

    async def check_and_set_idempotency(self, thread_id: str, webhook_msg_id: str, node_name: str, msg_index: int) -> bool:
        """
        Checks if a specific outgoing message for a given webhook payload has already been sent.
        Uses (thread_id + webhook_msg_id + node_name + msg_index) hash.
        Returns True if we should process (is new), False if already processed.
        """
        full_key = self._idempotency_key(thread_id, webhook_msg_id, node_name, msg_index)
        # Store for 24 hours
        is_new = await self.redis.set(full_key, "sent", nx=True, ex=86400)
        return bool(is_new)

    async def release_idempotency(self, thread_id: str, webhook_msg_id: str, node_name: str, msg_index: int) -> None:
        """
        Give the claim back when the send it was claiming never happened.

        The claim is taken BEFORE the outbound POST, which is right — two concurrent turns must not
        both send the same bubble. But with no way to undo it, a send that failed was recorded as
        sent: the TaskIQ retry then skipped it as a duplicate and the customer never received that
        message at all, permanently. Observed as an `httpx.ConnectTimeout` to Meta, which is a blip,
        not a reason to lose a reply.

        Best-effort by design — a Redis hiccup here must not turn a failed send into a raised
        exception on top of the one already being handled.
        """
        try:
            await self.redis.delete(self._idempotency_key(thread_id, webhook_msg_id, node_name, msg_index))
        except Exception as e:
            logger.warning(f"[{thread_id}] Could not release the idempotency claim: {e}")

    @staticmethod
    def _idempotency_key(thread_id: str, webhook_msg_id: str, node_name: str, msg_index: int) -> str:
        raw_string = f"{thread_id}:{webhook_msg_id}:{node_name}:{msg_index}"
        return f"idem_out:{hashlib.sha256(raw_string.encode('utf-8')).hexdigest()}"