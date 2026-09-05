"""
Public asset routes.

The brochure is served from this app rather than a file-sharing link because Meta's servers fetch
the URL themselves: they need a direct response with `application/pdf`, and a Google Drive "view"
link returns an HTML page instead. Serving it here also means the URL follows whatever tunnel or
domain the deployment already uses for its webhooks, with no second thing to keep in sync.
"""
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/brochure", include_in_schema=False)
async def get_brochure():
    """
    Serve the Otohom lookbook PDF.

    Deliberately unauthenticated: it is a marketing brochure that the company hands to anyone who
    asks, and Meta cannot present credentials when fetching an attachment. Nothing else is exposed —
    the path is fixed in config, never taken from the request, so this cannot be walked into other
    files.
    """
    path = Path(settings.BROCHURE_FILE_PATH)
    if not path.is_file():
        logger.error(f"Brochure requested but not found at {path}")
        raise HTTPException(status_code=404, detail="Brochure not available.")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=settings.BROCHURE_FILENAME,
        headers={"Cache-Control": "public, max-age=86400"},
    )
