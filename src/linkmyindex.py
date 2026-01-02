import os
import argparse
import re
import logging
import json
from rich.logging import RichHandler
from template_service import TemplateService
from gemini_service import GeminiService
import pdf_processor
import concurrent.futures
import shutil
import sys
import time
# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)
LOG_FILENAME = "logs/linkmyindex.log"

# Setup dual logging (Console with Rich, File with standard layout)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True, show_path=False),
        logging.FileHandler(LOG_FILENAME, mode='w')
    ]
)
# Update file handler formatter to include date/time/logger name
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.FileHandler):
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)

def parse_date(date_str):
    months = {
        "January": "01", "February": "02", "March": "03", "April": "04",
        "May": "05", "June": "06", "July": "07", "August": "08",
        "September": "09", "October": "10", "November": "11", "December": "12"
    }
    date_str = date_str.strip()
    match = re.search(r"(\d+)\s+([A-Za-z]+),\s+(\d{4})", date_str)
    if match:
        day = match.group(1).zfill(2)
        month_name = match.group(2)
        year = match.group(3)
        month = months.get(month_name)
        if month:
            return f"{year}-{month}-{day}"
    return None

def load_itemlist(path):
    d2id = {}
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                val = line.strip().strip('"')
                if not val:
                    continue
                match = re.search(r"(\d{4}-\d{2}-\d{2})", val)
                if match:
                    d2id[match.group(1)] = val
    except (FileNotFoundError, PermissionError) as error_msg:
        logger.error("Error loading %s: %s", path, error_msg)
    except Exception as error_msg:
        logger.error("Unexpected error loading %s: %s", path, error_msg)
    return d2id

def sanitize_text(text):
    if not text:
        return text
    rules = [
        (r"=", "-"),
        (r"\s+", " "),
    ]
    for pattern, replacement in rules:
        text = re.sub(pattern, replacement, text)
    return text.strip()

def process_index(pdf_path, prompt_content, itemlist_path="data/itemlist.txt", model_name="gemini-3-flash-preview", thought_level="low", workers=1, keep_temp=False):
    """
    Process the index PDF using Gemini, extract items, and link them to Archive.org IDs.
    Supports parallel processing by splitting the PDF into pages.

    Args:
        pdf_path (str): Path to the input PDF file.
        prompt_content (str): The prompt text to guide the Gemini model.
        itemlist_path (str): Path to the text file mapping dates to IDs.
        model_name (str): Name of the Gemini model to use.
        thought_level (str): The thinking level configuration for the model.
        workers (int): Number of parallel workers (1 for sequential/single-file).
        keep_temp (bool): If True, temporary pages are not deleted after processing.

    Returns:
        tuple: A tuple containing a list of processed items (dicts) and the usage metadata.
    """
    logger.info("Starting index processing for: %s (Workers: %d)", pdf_path, workers)
    date_to_id = load_itemlist(itemlist_path)
    
    raw_items = []
    usage_list = []

    if workers > 1:
        # Prallel processing strategy
        temp_dir = os.path.join("generated", "temp_pages")
        logger.info("Parallel processing enabled. Splitting PDF to %s...", temp_dir)
        
        try:
            # 1. Split PDF into single pages
            # Reuse pdf_processor logic. keep_ocr=False to force fresh OCR by Gemini (removes text layer)
            page_paths = pdf_processor.process_pdf(
                input_path=pdf_path, 
                pages_arg=None, 
                keep_ocr=True, 
                output_dir=temp_dir
            )
            
            # 2. Process pages in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                # Create a task for each page
                futures = []
                for p_path in page_paths:
                    futures.append(executor.submit(
                        _process_single_page_task, 
                        p_path, 
                        prompt_content, 
                        model_name, 
                        thought_level
                    ))
                
                for future in futures:
                    try:
                        items, u_data = future.result()
                        raw_items.extend(items)
                        if u_data:
                            usage_list.append(u_data)
                    except Exception as exc:
                        logger.error("A page generated an exception: %s", exc)

        finally:
            # Cleanup temp files
            if not keep_temp and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.debug("Cleaned up temporary directory: %s", temp_dir)
            elif keep_temp:
                 logger.info("Temporary files kept at: %s", temp_dir)

    else:
        # Single file processing (Legacy/Sequential)
        service = GeminiService(model_name=model_name)
        raw_items, usage = service.transcribe_index(pdf_path, prompt=prompt_content, thought_level=thought_level)
        if usage:
            usage_list.append(usage)

    # Aggregate usage stats
    total_input = sum(u.prompt_token_count for u in usage_list)
    total_output = sum(u.candidates_token_count for u in usage_list)
    total_tokens = sum(u.total_token_count for u in usage_list)
    
    # Create a dummy usage object for return
    class SimpleUsageMetadata:
        def __init__(self, prompt_token_count, candidates_token_count, total_token_count):
            self.prompt_token_count = prompt_token_count
            self.candidates_token_count = candidates_token_count
            self.total_token_count = total_token_count

    aggregated_usage = SimpleUsageMetadata(
        prompt_token_count=total_input,
        candidates_token_count=total_output,
        total_token_count=total_tokens
    )
    
    processed_items = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
            
        topic = sanitize_text(item.get("topic", "General Index"))
        subhead = sanitize_text(item.get("subhead", ""))
        author = sanitize_text(item.get("author", "Anon"))
        date_raw = sanitize_text(item.get("date", ""))
        page_num = str(item.get("page", ""))
        index_page = str(item.get("index_page", ""))
        
        if not date_raw or not page_num:
            logger.warning("Skipping item with missing date or page: %s", item)
            continue

        processed_item = {
            "topic": topic,
            "subhead": subhead,
            "author": author,
            "date": date_raw,
            "page": page_num,
            "index_page": index_page,
            "link": ""
        }
        
        date_iso = parse_date(date_raw)
        if date_iso and date_iso in date_to_id:
            processed_item["link"] = f"https://archive.org/details/{date_to_id[date_iso]}/page/{page_num}/mode/1up"
        
        processed_items.append(processed_item)

    logger.info("Processed %d items successfully.", len(processed_items))
    return processed_items, aggregated_usage

def _process_single_page_task(pdf_path, prompt, model, thought_level):
    """Helper function to process a single page in a thread."""
    service = GeminiService(model_name=model)
    # Disable Live progress for parallel tasks to avoid console mess
    return service.transcribe_index(pdf_path, prompt=prompt, thought_level=thought_level, show_progress=False)

def handle_batch_mode(args, parser):
    """Handles the batch mode workflow."""
    
    # Ensure output directory exists before checking/creating batch.json
    os.makedirs(args.output_dir, exist_ok=True)
    batch_file = os.path.join(args.output_dir, "batch.json")
    service = GeminiService(model_name=args.model)
    
    if os.path.exists(batch_file):
        logger.info("Found existing batch file: %s", batch_file)
        try:
            with open(batch_file, "r") as f:
                batch_info = json.load(f)
        except Exception as e:
            logger.error("Failed to read batch file: %s", e)
            return

        job_name = batch_info.get("job_name")
        if not job_name:
            logger.error("Invalid batch file: missing job_name")
            return
            
        try:
            job = service.get_batch_job(job_name)
            logger.info("Batch Job Status: %s", job.state)
            
            if job.state == "JOB_STATE_RUNNING":
                logger.info("Batch job is still running. Please check back later.")
                return
            elif job.state.name == "JOB_STATE_SUCCEEDED":
                logger.info("Batch job completed. Retrieving results...")
                
                # Extract items from batch job using GeminiService
                raw_items = service.extract_batch_results(job)
                
                if not raw_items:
                    logger.error("No items extracted from batch job")
                    return
                
                # Load itemlist
                date_to_id = load_itemlist(args.itemlist)
                
                # Process items (same logic as in process_index)
                processed_items = []
                for item in raw_items:
                    if not isinstance(item, dict):
                        continue
                        
                    topic = sanitize_text(item.get("topic", "General Index"))
                    subhead = sanitize_text(item.get("subhead", ""))
                    author = sanitize_text(item.get("author", "Anon"))
                    date_raw = sanitize_text(item.get("date", ""))
                    page_num = str(item.get("page", ""))
                    index_page = str(item.get("index_page", ""))
                    
                    if not date_raw or not page_num:
                        logger.warning("Skipping item with missing date or page: %s", item)
                        continue

                    processed_item = {
                        "topic": topic,
                        "subhead": subhead,
                        "author": author,
                        "date": date_raw,
                        "page": page_num,
                        "index_page": index_page,
                        "link": ""
                    }
                    
                    date_iso= parse_date(date_raw)
                    if date_iso and date_iso in date_to_id:
                        processed_item["link"] = f"https://archive.org/details/{date_to_id[date_iso]}/page/{page_num}/mode/1up"
                    
                    processed_items.append(processed_item)
                
                logger.info("Processed %d items successfully.", len(processed_items))
                
                # Save to JSON
                json_output_path = os.path.join(args.output_dir, args.json_file if args.json_file else "batch_results.json")
                with open(json_output_path, "w", encoding="utf-8") as f_json:
                    json.dump(processed_items, f_json, indent=4, ensure_ascii=False)
                logger.info("Extracted data saved to: %s", json_output_path)
                
                # Generate HTML
                if not args.skip_index_creation:
                    template_svc = TemplateService(template_dir="templates")
                    html_content = template_svc.render_index(processed_items, template_name=args.template)
                    
                    output_path = os.path.join(args.output_dir, "index.html")
                    template_svc.save_report(html_content, output_path)
                    
                    logger.info("Successfully generated index.html at %s", output_path)
                else:
                    logger.info("Skipping index HTML generation as requested.")
                
                # Clean up batch.json
                batch_file = os.path.join(args.output_dir, "batch.json")
                if os.path.exists(batch_file):
                    os.remove(batch_file)
                    logger.info("Cleaned up batch tracking file")
                    service.delete_batch(name=job.name)
                
                return
            elif job.state == "JOB_STATE_FAILED":
                logger.error("Batch job failed: %s", getattr(job, 'error', 'Unknown error'))
                return
            elif job.state == "JOB_STATE_PENDING":
                logger.info("Batch job not yet started: %s", job.state)
                return
            else:
                logger.warning("Batch job in unexpected state: %s", job.state)
                return
        except Exception as e:
            logger.error("Error checking batch job: %s", e)
            return

    else:
        logger.info("No existing batch file found using --batch-mode. Initiating new batch job...")
        
        # 1. Read prompt
        if not os.path.exists(args.prompt_file):
            logger.critical("Prompt file not found: %s", args.prompt_file)
            return
        
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_content = f.read()

        # 2. Split PDF
        temp_dir = os.path.join("generated", "temp_pages_batch")
        
        try:
            page_paths = pdf_processor.process_pdf(
                input_path=args.index_file, 
                pages_arg=None, 
                keep_ocr=True,
                output_dir=temp_dir
            )
            
            # 3. Prepare requests
            requests = []
            for p_path in page_paths:
                req, _ = service.prepare_batch_request(p_path, prompt_content, args.thought_level)
                requests.append(req)
            
            # 4. Submit Batch
            job = service.create_batch_job(requests)
            
            # Log batch job configuration
            logger.info("Batch Job Configuration:")
            logger.info("  Model: %s", args.model)
            logger.info("  Thought Level: %s", args.thought_level)
            logger.info("  Input File: %s", args.index_file)
            logger.info("  Total Pages: %d", len(page_paths))
            logger.info("  Workers: Batch API (async)")
            
            # 5. Save batch.json
            batch_info = {
                "job_name": job.name,
                "created_at": time.time(),
                "original_file": args.index_file,
                "model": args.model,
                "thought_level": args.thought_level,
                "page_count": len(page_paths)
            }
            with open(batch_file, "w") as f:
                json.dump(batch_info, f, indent=4)
                
            logger.info("Batch job started! Details saved to %s", batch_file)
            logger.info("Run this command again later to check status and retrieve results.")
            
        except Exception as e:
            logger.exception("Failed to start batch job: %s", e)
        finally:
             if not args.keep_temporary_files and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.debug("Cleaned up temp files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate linked index HTML from PDF using Gemini Service.")
    parser.add_argument("--index-file", "-i", required=True, help="Input PDF index file")
    parser.add_argument("--itemlist", "-l", default="data/itemlist.txt", help="Path to itemlist.txt mapping, default: data/itemlist.txt")
    parser.add_argument("--template", "-t", default="index.html.j2", help="Template file name in templates, default: index.html.j2")
    parser.add_argument("--prompt-file", "-p", default="prompts/mr.txt", help="Path to file containing the prompt for Gemini, default: prompts/mr.txt")
    parser.add_argument("--model", "-m", default="gemini-3-flash-preview", 
                        choices=["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3-pro-preview"],
                        help="Gemini model to use (default: gemini-3-flash-preview)")
    parser.add_argument("--thought-level", default="low",
                        choices=["low", "medium", "high", "minimal"],
                        help="Thinking level for Gemini models (default: low)")
    parser.add_argument("--output-dir", "-o", default="generated", help="Directory to save the generated HTML (default: generated)")
    parser.add_argument("--json-file", "-j", default=None, help="Filename to save the raw extracted data (default: derived from input filename)")
    parser.add_argument("--force-ocr", "-f", action="store_true", help="Force re-running OCR extraction even if local JSON exists")
    parser.add_argument("--verbose", "-v", action="store_true", help="Increase output verbosity")
    parser.add_argument("--skip-index-creation", "-si", action="store_true", help="Skip the index HTML generation part")
    parser.add_argument("--workers", "-w", type=int, default=10, help="Number of parallel workers for page processing (default: 10)")
    parser.add_argument("--keep-temporary-files", "-k", action="store_true", help="Keep temporary split pages after processing")
    parser.add_argument("--batch-mode", "-b", action="store_true", help="Use Gemini Batch API for processing")
    args = parser.parse_args()

    if args.json_file is None:
        input_basename = os.path.basename(args.index_file)
        filename_without_ext = os.path.splitext(input_basename)[0]
        args.json_file = f"{filename_without_ext}.json"

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    if args.batch_mode:
        handle_batch_mode(args, parser)
        # Exit after handling batch mode (whether submitted or checked)
        sys.exit(0)

    try:
        # Ensure output directory exists
        if args.output_dir != ".":
            os.makedirs(args.output_dir, exist_ok=True)
            logger.debug("Ensured output directory exists: %s", args.output_dir)

        # Read prompt file
        if not os.path.exists(args.prompt_file):
            logger.critical("Prompt file not found: %s", args.prompt_file)
            raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
        
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_content = f.read()

        # Check if extracted JSON already exists
        json_output_path = os.path.join(args.output_dir, args.json_file)
        
        results_final = []
        usage_data = None
        
        if os.path.exists(json_output_path) and not args.force_ocr:
            logger.info("Found existing extracted data at %s. Skipping Gemini OCR processing.", json_output_path)
            try:
                with open(json_output_path, "r", encoding="utf-8") as f_in:
                    results_final = json.load(f_in)
                logger.info("Loaded %d items from local cache.", len(results_final))
            except json.JSONDecodeError as e:
                logger.error("Failed to load existing JSON file: %s. Re-running processing.", e)
                # Fallback to processing if file is corrupted
                results_final, usage_data = process_index(
                    pdf_path=args.index_file,
                    prompt_content=prompt_content,
                    itemlist_path=args.itemlist,
                    model_name=args.model,
                    thought_level=args.thought_level,
                    workers=args.workers,
                    keep_temp=args.keep_temporary_files
                )
                # Save processed data to JSON locally
                with open(json_output_path, "w", encoding="utf-8") as f_json:
                    json.dump(results_final, f_json, indent=4, ensure_ascii=False)
                logger.info("Extracted data saved to: %s", json_output_path)
        else:
            results_final, usage_data = process_index(
                pdf_path=args.index_file,
                prompt_content=prompt_content,
                itemlist_path=args.itemlist,
                model_name=args.model,
                thought_level=args.thought_level,
                workers=args.workers,
                keep_temp=args.keep_temporary_files
            )
            
            # Save processed data to JSON locally
            with open(json_output_path, "w", encoding="utf-8") as f_json:
                json.dump(results_final, f_json, indent=4, ensure_ascii=False)
            logger.info("Extracted data saved to: %s", json_output_path)

        # Generate HTML using TemplateService
        if not args.skip_index_creation:
            template_svc = TemplateService(template_dir="templates")
            html_content = template_svc.render_index(results_final, template_name=args.template)
            
            output_path = os.path.join(args.output_dir, "index.html")
            template_svc.save_report(html_content, output_path)
            
            logger.info("Successfully generated index.html at %s using model '%s'.", output_path, args.model)
        else:
            logger.info("Skipping index HTML generation as requested.")
        
        if usage_data:
            # Instantiate service temporarily to use its cost calculation logic
            temp_service = GeminiService(model_name=args.model, price_table_path="price_table.json")
            est_cost_usd = temp_service.calculate_cost(usage_data.prompt_token_count, usage_data.candidates_token_count)
            usd_to_eur = temp_service.pricing_config.get("usd_to_eur", 0.96)
            est_cost_eur = est_cost_usd * usd_to_eur
            
            logger.info("Final Token Usage Summary: Total: %d, Prompt: %d, Candidates: %d. Estimated Cost: €%.6f",
                        usage_data.total_token_count,
                        usage_data.prompt_token_count,
                        usage_data.candidates_token_count,
                        est_cost_eur)
    except Exception as e:
        logger.exception("A fatal error occurred: %s", e)
