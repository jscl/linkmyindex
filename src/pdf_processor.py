"""
Module for processing PDF files (page selection, redaction).
"""

import logging
import os

import pymupdf

logger = logging.getLogger(__name__)

def parse_pages(pages_arg: str) -> list[int]:
    """
    Parses a string of page ranges (e.g. "1,3-5") into a list of 0-based page indices.
    """
    pages = set()
    parts = pages_arg.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            start, end = int(start), int(end)
            pages.update(range(start - 1, end))
        else:
            pages.add(int(part) - 1)
    return sorted(list(pages))

def process_pdf(input_path: str, pages_arg: str | None, keep_ocr: bool = True, output_dir: str = './generated/temp_pages') -> list[str]:
    """
    Processes a PDF: selects pages (unless keep_ocr is True).
    Returns the path to the processed PDF.
    """
    logger.info("Processing PDF: %s", input_path)
    doc = pymupdf.open(input_path)
    
    # Select pages
    if pages_arg:
        selected_pages = parse_pages(pages_arg)
        # Validate page numbers
        max_page = len(doc) - 1
        selected_pages = [p for p in selected_pages if 0 <= p <= max_page]
        if not selected_pages:
            raise ValueError("No valid pages selected.")
        doc.select(selected_pages)
        logger.info("Selected %d pages: %s", len(doc), ", ".join(str(int(p + 1)) for p in selected_pages))

    output_paths = []
    base_output_filename = os.path.basename(input_path)
    filename_stem, filename_ext = os.path.splitext(base_output_filename)
    
    # Apply redactions if needed (once for the whole doc, before splitting)
    # Note: If we save individual pages, we need to ensure redactions are applied to them.
    # We can modify the doc in memory and then extract pages.
    if not keep_ocr:
        # Use redaction to remove text but keep images and graphics
        logger.info("Applying redactions to remove text/OCR layer...")
        for page in doc:
            page.add_redact_annot(page.rect)
            page.apply_redactions(
                images=0, #pymupdf.PDF_REDACT_IMAGE_NONE,  # fail-safe: keep images
                graphics=0#pymupdf.PDF_REDACT_LINE_ART_NONE  # fail-safe: keep vector graphics
            )
            
    # Split pages into individual files
    split_dir = output_dir
    os.makedirs(split_dir, exist_ok=True)
    
    logger.info("Splitting PDF into individual pages...")
    for i in range(len(doc)):
        page_num = i + 1
        # Create a new document for the single page
        new_doc = pymupdf.open()
        new_doc.insert_pdf(doc, from_page=i, to_page=i)
        
        page_filename = f"{filename_stem}_page_{page_num}{filename_ext}"
             
        page_output_path = os.path.join(split_dir, page_filename)
        
        #new_doc.ez_save(page_output_path)
        new_doc.save(page_output_path)
        new_doc.close()
        output_paths.append(page_output_path)
        logger.debug("Saved page %d to: %s", page_num, page_output_path)

    doc.close()
    logger.info("Split PDF into %d files.", len(output_paths))
    return output_paths
