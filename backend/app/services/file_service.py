import asyncio
import io
import logging

import fitz  # PyMuPDF
from fastapi import HTTPException, UploadFile
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import FileContext

logger = logging.getLogger(__name__)

ALLOWED_MIME_TYPES = {
    "application/pdf": "pdf",
    "image/jpeg": "jpg",
    "image/png": "png",
}


async def validate_upload(
    file: UploadFile, settings
) -> tuple[bytes, str]:
    """Validate file type and size. Returns (file_bytes, extension)."""
    content_type = file.content_type or ""
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Allowed: PDF, JPG, PNG.",
        )

    file_bytes = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB.",
        )

    return file_bytes, ALLOWED_MIME_TYPES[content_type]


def pdf_to_images(file_bytes: bytes, max_pages: int) -> list[Image.Image]:
    """Convert PDF pages to PIL Images using PyMuPDF at 150 DPI."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    if doc.page_count > max_pages:
        doc.close()
        raise HTTPException(
            status_code=400,
            detail=f"PDF has {doc.page_count} pages. Maximum allowed: {max_pages}.",
        )

    images = []
    matrix = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


async def extract_file_content(images: list[Image.Image], app_state) -> str:
    """Use Gemini Vision to extract all text and data from document images."""
    prompt = (
        "Extract ALL text, tables, numerical values, and data from this medical document. "
        "Preserve the original structure (headings, tables, lists). "
        "Include all test names, values, units, and reference ranges if present. "
        "Do NOT summarize — extract everything exactly as shown. "
        "Do NOT translate — keep the text in the SAME language as the original document."
    )

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: app_state.gemini.generate_content([prompt, *images]),
    )
    if not response.candidates or not response.text:
        raise ValueError("Empty response from Gemini Vision")
    return response.text.strip()


async def save_file_context(
    session_id: str,
    filename: str,
    file_type: str,
    extracted_text: str,
    page_count: int,
    db_session: AsyncSession,
) -> FileContext:
    """Save extracted file content to the database."""
    row = FileContext(
        session_id=session_id,
        original_filename=filename,
        file_type=file_type,
        extracted_text=extracted_text,
        page_count=page_count,
    )
    db_session.add(row)
    await db_session.commit()
    await db_session.refresh(row)
    return row


async def get_latest_file_context(
    session_id: str, db_session: AsyncSession
) -> str | None:
    """Get the most recent extracted text for a session. Returns None if no file."""
    result = await db_session.execute(
        select(FileContext.extracted_text)
        .where(FileContext.session_id == session_id)
        .order_by(FileContext.created_at.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    return row


async def get_file_list(session_id: str, db_session: AsyncSession) -> list[FileContext]:
    """Get all uploaded files for a session, newest first."""
    result = await db_session.execute(
        select(FileContext)
        .where(FileContext.session_id == session_id)
        .order_by(FileContext.created_at.desc())
    )
    return list(result.scalars().all())
