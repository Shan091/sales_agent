import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TranscoderService:
    """
    Speech-to-text for WhatsApp voice notes. Not implemented yet.

    `transcribe_audio` returns None rather than a placeholder string. That distinction matters:
    a stand-in transcript is indistinguishable downstream from something the customer actually
    said, so the agent would answer a question nobody asked and — in autonomy mode — could act
    on it. Returning None lets the caller say plainly that voice isn't supported yet and ask
    for text, which is honest and keeps the conversation moving.
    """

    @staticmethod
    async def transcribe_audio(media_id: Optional[str]) -> Optional[str]:
        logger.info(f"Voice note received (media_id={media_id}) but transcription is not enabled.")
        return None
