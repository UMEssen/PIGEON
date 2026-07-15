"""
Stage 8: PDF document processing with Docling.

This module wraps the Docling library for dual-mode PDF processing:

- **Standard mode** (fast): uses Docling's built-in PDF parser with
  pypdfium2.  Works well for digitally-born PDFs that already contain
  a text layer.
- **VLM mode** (fallback): when the standard parser produces little or
  no text (i.e. the PDF is a scanned image), the processor falls back
  to a vision-language model (e.g. Qwen2.5-VL) served via a vLLM
  endpoint to perform OCR on each page image.

The main entry point is ``DoclingProcessor.parse_pdf()``, which returns
the extracted HTML, page images, a flag indicating whether the PDF had
a native text layer, and metadata about the document.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (
    LLM_ENDPOINT_VLM,
    LLM_MODEL_VLM,
)

import io
import re
import logging
import tempfile
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DoclingProcessor:
    """Dual-mode PDF processor: standard text extraction + VLM OCR fallback.

    Parameters
    ----------
    vlm_endpoint : str
        OpenAI-compatible chat/completions URL for the VLM.
        Defaults to ``config.LLM_ENDPOINT_VLM``.
    vlm_model : str
        Model name served at ``vlm_endpoint``.
        Defaults to ``config.LLM_MODEL_VLM``.
    min_text_length : int
        Minimum character count to consider the PDF as "having text".
        If the standard parser produces fewer characters the VLM fallback
        is triggered automatically (unless ``force_vlm`` is set).
    """

    def __init__(
        self,
        vlm_endpoint: str = LLM_ENDPOINT_VLM,
        vlm_model: str = LLM_MODEL_VLM,
        min_text_length: int = 50,
    ):
        self.vlm_endpoint = vlm_endpoint
        self.vlm_model = vlm_model
        self.min_text_length = min_text_length

        # Lazy-loaded Docling converters (heavy imports)
        self._standard_converter = None
        self._vlm_converter = None

    # ------------------------------------------------------------------
    # Lazy initialisation of Docling converters
    # ------------------------------------------------------------------

    def _get_standard_converter(self):
        """Return the standard Docling ``DocumentConverter`` (lazy init)."""
        if self._standard_converter is None:
            from docling.document_converter import DocumentConverter
            self._standard_converter = DocumentConverter()
            logger.info("Standard Docling converter initialised")
        return self._standard_converter

    def _get_vlm_converter(self):
        """Return a Docling converter configured for VLM-based OCR (lazy init).

        This converter sends page images to the VLM endpoint for
        transcription.  It is only used when standard text extraction
        fails (scanned documents).
        """
        if self._vlm_converter is None:
            try:
                from docling.document_converter import DocumentConverter, PdfFormatOption
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.pipeline.vlm_pipeline import VlmPipelineOptions
                from docling.datamodel.pipeline_options import (
                    VlmModelOptions,
                    ApiVlmModel,
                )

                # Configure the VLM pipeline to call our vLLM endpoint
                vlm_options = VlmPipelineOptions(
                    vlm_options=VlmModelOptions(
                        model=ApiVlmModel(
                            api_base=self.vlm_endpoint,
                            model_name=self.vlm_model,
                        )
                    )
                )

                pipeline_options = PdfPipelineOptions(vlm_options=vlm_options)

                self._vlm_converter = DocumentConverter(
                    format_options={
                        "pdf": PdfFormatOption(pipeline_options=pipeline_options),
                    }
                )
                logger.info("VLM Docling converter initialised (endpoint=%s)", self.vlm_endpoint)
            except ImportError as exc:
                logger.warning(
                    "VLM pipeline not available (missing docling extras): %s", exc
                )
                # Fall back to the standard converter
                self._vlm_converter = self._get_standard_converter()

        return self._vlm_converter

    # ------------------------------------------------------------------
    # Main public API
    # ------------------------------------------------------------------

    def parse_pdf(
        self,
        pdf_path: str,
        force_vlm: bool = False,
    ) -> Tuple[str, List[Any], bool, Dict[str, Any]]:
        """Parse a PDF and return extracted content.

        Parameters
        ----------
        pdf_path : str or Path
            Path to the PDF file on disk.
        force_vlm : bool
            When ``True``, skip the standard parser and go straight to
            the VLM pipeline (useful for known scanned documents).

        Returns
        -------
        html : str
            HTML representation of the document produced by Docling.
        images : list
            List of page images (PIL Image objects) extracted from the PDF.
        has_text : bool
            ``True`` if the PDF had a usable native text layer.
        metadata : dict
            Extra metadata: page count, converter used, timings, etc.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        metadata: Dict[str, Any] = {
            "filename": pdf_path.name,
            "page_count": 0,
            "converter_used": "standard",
        }

        # Extract page images (always useful for the UI preview)
        images = self._extract_page_images(pdf_path)
        metadata["page_count"] = len(images)

        # Step 1: try standard (fast) conversion unless VLM is forced
        html = ""
        has_text = False

        if not force_vlm:
            try:
                converter = self._get_standard_converter()
                result = converter.convert(str(pdf_path))

                # Docling result -> export to HTML and markdown
                html = result.document.export_to_html()
                markdown = result.document.export_to_markdown()
                raw_text = result.document.export_to_text()

                plain_text = self.extract_plain_text(html, markdown, raw_text)
                has_text = len(plain_text.strip()) >= self.min_text_length

                if has_text:
                    metadata["converter_used"] = "standard"
                    logger.info(
                        "Standard parser extracted %d chars from %s",
                        len(plain_text),
                        pdf_path.name,
                    )
                    return html, images, has_text, metadata

                logger.info(
                    "Standard parser got only %d chars -- falling back to VLM",
                    len(plain_text),
                )
            except Exception as exc:
                logger.warning("Standard conversion failed: %s", exc)

        # Step 2: VLM fallback (scanned / image-only PDF)
        try:
            converter = self._get_vlm_converter()
            result = converter.convert(str(pdf_path))

            html = result.document.export_to_html()
            metadata["converter_used"] = "vlm"
            has_text = True  # VLM always produces text from images
            logger.info("VLM converter processed %s", pdf_path.name)
        except Exception as exc:
            logger.error("VLM conversion also failed: %s", exc)
            metadata["converter_used"] = "failed"
            has_text = False

        return html, images, has_text, metadata

    # ------------------------------------------------------------------
    # Text extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_plain_text(
        html: str = "",
        markdown: str = "",
        raw_text: str = "",
    ) -> str:
        """Combine and clean the various Docling text outputs.

        Docling can produce HTML, Markdown, and plain text.  This helper
        picks the richest non-empty representation and strips tags to
        return clean plain text suitable for the extraction model.
        """
        # Prefer raw_text if available and substantial
        if raw_text and len(raw_text.strip()) > 50:
            return raw_text.strip()

        # Fall back to markdown (strip simple formatting)
        if markdown and len(markdown.strip()) > 50:
            text = markdown
            # Remove markdown headings, bold, italic markers
            text = re.sub(r"#{1,6}\s*", "", text)
            text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
            text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
            return text.strip()

        # Fall back to HTML with tag stripping
        if html:
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        return ""

    @staticmethod
    def inject_captions_into_html(
        html: str,
        captions: Dict[int, str],
    ) -> str:
        """Insert VLM-generated captions for images into the HTML.

        Parameters
        ----------
        html : str
            The Docling HTML output.
        captions : dict
            Mapping of image index to caption text.

        Returns
        -------
        str
            Updated HTML with ``<figcaption>`` elements added after
            ``<img>`` tags.
        """
        if not captions:
            return html

        for idx, caption in sorted(captions.items()):
            # Find the idx-th <img> tag and add a figcaption after it
            img_tags = list(re.finditer(r"(<img[^>]*>)", html))
            if idx < len(img_tags):
                match = img_tags[idx]
                insert_pos = match.end()
                figcaption = f'<figcaption class="vlm-caption">{caption}</figcaption>'
                html = html[:insert_pos] + figcaption + html[insert_pos:]

        return html

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_page_images(pdf_path: Path) -> List[Any]:
        """Render each page of the PDF as a PIL Image.

        Uses ``pypdfium2`` which is already a dependency of Docling.
        Returns an empty list if rendering fails.
        """
        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(str(pdf_path))
            images = []
            for page_idx in range(len(pdf)):
                page = pdf[page_idx]
                # Render at 150 DPI for reasonable quality / size trade-off
                bitmap = page.render(scale=150 / 72)
                pil_image = bitmap.to_pil()
                images.append(pil_image)
            return images
        except Exception as exc:
            logger.warning("Could not extract page images: %s", exc)
            return []
