import os
import json
import logging
from google import genai
from google.genai import types
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel

# Configure logger for this module
logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self, api_key=None, model_name="gemini-3-flash-preview", price_table_path="price_table.json"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.error("GEMINI_API_KEY environment variable not set.")
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.console = Console()
        self.price_table_path = price_table_path
        self.pricing_config = self._load_pricing()

    def _load_pricing(self):
        config = {}
        if os.path.exists(self.price_table_path):
            try:
                with open(self.price_table_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception as e:
                logger.error("Failed to load pricing table from %s: %s", self.price_table_path, e)
        else:
            logger.warning("Pricing table not found at %s", self.price_table_path)
        return config

    def calculate_cost(self, prompt_tokens, candidate_tokens):
        pricing = self.pricing_config.get("pricing", {})
        
        # Try exact match first
        rates = pricing.get(self.model_name)
        
        # Fallback to "partial" match if exact logic fails, or default to flash
        if not rates:
            if "flash" in self.model_name.lower():
                rates = pricing.get("gemini-2.5-flash") # Fallback to standard flash
            else:
                rates = pricing.get("gemini-2.5-pro") # Fallback to standard pro
        
        # If still nothing, use safety default (Flash rates)
        if not rates:
             rates = {"input": 0.075, "output": 0.30}

        input_rate = rates.get("input", 0.0)
        output_rate = rates.get("output", 0.0)

        # Check for tiered pricing (specifically for Gemini 3 Pro Preview > 200k context)
        if "tiered" in rates:
            tier_info = rates["tiered"]
            threshold = tier_info.get("threshold", 200000)
            if prompt_tokens > threshold:
                input_rate = tier_info.get("input_high", input_rate)
                output_rate = tier_info.get("output_high", output_rate)
        
        input_cost = (prompt_tokens / 1_000_000) * input_rate
        output_cost = (candidate_tokens / 1_000_000) * output_rate
        return input_cost + output_cost

    def upload_file(self, file_path):
        import shutil
        import uuid
        
        # Check for non-ASCII characters in filename
        filename = os.path.basename(file_path)
        is_ascii = all(ord(char) < 128 for char in filename)
        
        if not is_ascii:
            logger.info("Detected non-ASCII characters in filename '%s'. Creating temporary ASCII copy.", filename)
            ext = os.path.splitext(filename)[1]
            # Create a safe temp filename in the same directory
            safe_filename = f"temp_{uuid.uuid4().hex}{ext}"
            safe_path = os.path.join(os.path.dirname(file_path), safe_filename)
            
            try:
                shutil.copy(file_path, safe_path)
                logger.info("Uploading %s to Gemini...", safe_path)
                # Helper allows auto-deletion of temp file via finally block
                return self.client.files.upload(file=safe_path)
            finally:
                if os.path.exists(safe_path):
                    try:
                        os.remove(safe_path)
                        logger.debug("Removed temporary file %s", safe_path)
                    except OSError as e:
                        logger.warning("Failed to remove temporary file %s: %s", safe_path, e)
        else:
            logger.info("Uploading %s to Gemini...", file_path)
            return self.client.files.upload(file=file_path)

    def delete_file(self, file_name):
        try:
            self.client.files.delete(name=file_name)
            logger.debug("Deleted file %s from Gemini", file_name)
        except Exception as e:
            logger.warning("Cleanup warning: Could not delete %s: %s", file_name, e)

    def transcribe_index(self, file_path, prompt, thought_level="low", show_progress=True):
        uploaded_file = self.upload_file(file_path)

        logger.info("Requesting transcription using model: %s (thought_level: %s)", self.model_name, thought_level)
        
        # Configure generation parameters
        if "gemini-3" in self.model_name:
            level_map = {
                "low": types.ThinkingLevel.LOW,
                "medium": types.ThinkingLevel.MEDIUM,
                "high": types.ThinkingLevel.HIGH,
                "minimal": types.ThinkingLevel.MINIMAL
            }
            level = level_map.get(thought_level.lower(), types.ThinkingLevel.LOW)
            config = types.GenerateContentConfig(
                temperature=0.0,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=level
                )
            )
        else:
            # For non-thinking models (e.g. Gemini 2.5), just set low temperature
            config = types.GenerateContentConfig(temperature=0.0)

        all_thoughts = []
        all_text = []

        usage = None
        full_raw_response = ""
        
        try:
            # Helper generator to yield chunks from the stream, updating Live if enabled
            stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=[uploaded_file, prompt],
                config=config
            )

            if show_progress:
                with Live(Panel("Waiting for Gemini...", title="[bold blue]Gemini Processing"), console=self.console, refresh_per_second=4) as live:
                    for chunk in stream:
                        if chunk.usage_metadata:
                            usage = chunk.usage_metadata

                        if not chunk.candidates:
                            continue
                            
                        for part in chunk.candidates[0].content.parts:
                            # In this SDK, part.thought is a bool indicating if the text is a thought
                            is_thought = getattr(part, 'thought', False)
                            txt = getattr(part, 'text', "")
                            
                            if is_thought and txt:
                                all_thoughts.append(str(txt))
                            elif txt:
                                all_text.append(str(txt))
                        
                        # Update status in the Live panel
                        thought_summary = "\n".join(all_thoughts)[-400:] # Show more thoughts
                        
                        # Estimate items by counting "topic": in the accumulated text
                        accumulated_text = "".join(all_text)
                        item_est = accumulated_text.count('"topic":')

                        live.update(Panel(
                            Group(
                                f"[bold green]Status:[/bold green] Receiving transcription stream... ({item_est} items identified)",
                                "",
                                f"[bold cyan]Gemini's Internal Thoughts (Streaming):[/bold cyan]\n[italic]{thought_summary}[/italic]"
                            ),
                            title=f"[bold white on blue] Gemini ({self.model_name}) thinking level: {thought_level} [/]",
                            border_style="blue",
                            padding=(1, 2)
                        ))
            else:
                # No Live display, just iterate
                for chunk in stream:
                    if chunk.usage_metadata:
                        usage = chunk.usage_metadata

                    if not chunk.candidates:
                        continue
                        
                    for part in chunk.candidates[0].content.parts:
                        is_thought = getattr(part, 'thought', False)
                        txt = getattr(part, 'text', "")
                        
                        if is_thought and txt:
                            all_thoughts.append(str(txt))
                        elif txt:
                            all_text.append(str(txt))

            # Live has exited here, logs will be visible
            if usage:
                est_cost_usd = self.calculate_cost(usage.prompt_token_count, usage.candidates_token_count)
                usd_to_eur = self.pricing_config.get("usd_to_eur", 0.96)
                est_cost_eur = est_cost_usd * usd_to_eur
                logger.info("Token usage - Total: %d, Prompt: %d, Candidates: %d. Estimated Cost: €%.6f", 
                            usage.total_token_count, 
                            usage.prompt_token_count, 
                            usage.candidates_token_count,
                            est_cost_eur)

            full_raw_response = "".join(all_text)
            
            # Show final thoughts summary
            if all_thoughts:
                self.console.print("\n[bold cyan]Gemini's Reasoning Summary:[/bold cyan]\n" + "\n".join(all_thoughts) + "\n")

            json_text = full_raw_response
            if "```json" in full_raw_response:
                json_text = full_raw_response.split("```json")[1].split("```")[0].strip()
            elif "```" in full_raw_response:
                json_text = full_raw_response.split("```")[1].split("```")[0].strip()
            else:
                # Fallback: try to find the start and end of the JSON array
                start_idx = full_raw_response.find("[")
                end_idx = full_raw_response.rfind("]")
                if start_idx != -1 and end_idx != -1:
                    json_text = full_raw_response[start_idx:end_idx+1].strip()

            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                # If that fails, maybe it's just a single object or something went wrong
                # Try to strip anything before first '[' if it exists
                logger.debug("First JSON parse failed, attempting strict list extraction.")
                start_idx = json_text.find("[")
                if start_idx != -1:
                    json_text = json_text[start_idx:]
                    # Try to find the matching last ']'
                    end_idx = json_text.rfind("]")
                    if end_idx != -1:
                        json_text = json_text[:end_idx+1]
                        data = json.loads(json_text)
                    else:
                        raise
                else:
                    raise
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
                data = data[0]
            
            logger.info("Successfully transcribed %d items", len(data) if isinstance(data, list) else 0)
            return (data if isinstance(data, list) else []), usage
        
        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON from Gemini: %s", e)
            logger.debug("Raw response text: %s", full_raw_response)
            return [], usage
        except Exception as e:
            logger.exception("An error occurred during transcription: %s", e)
            return [], usage
        finally:
            self.delete_file(uploaded_file.name)

    def prepare_batch_request(self, file_path, prompt, thought_level="low"):
        """Prepares a single request object for batch processing."""
        uploaded_file = self.upload_file(file_path)
        
        # Configure generation parameters
        if "gemini-3" in self.model_name:
            level_map = {
                "low": types.ThinkingLevel.LOW,
                "medium": types.ThinkingLevel.MEDIUM,
                "high": types.ThinkingLevel.HIGH,
                "minimal": types.ThinkingLevel.MINIMAL
            }
            level = level_map.get(thought_level.lower(), types.ThinkingLevel.LOW)
            config = types.GenerateContentConfig(
                temperature=0.0,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=level
                )
            )
        else:
             config = types.GenerateContentConfig(temperature=0.0)

        request = types.InlinedRequest(
            model=self.model_name,
            contents=[uploaded_file, prompt],
            config=config
        )
        return request, uploaded_file

    def create_batch_job(self, requests):
        """Creates a batch job with the given list of requests."""
        logger.info("Submitting batch job with %d requests...", len(requests))
        try:
            batch_job = self.client.batches.create(
                model=self.model_name,
                src=requests
            )
            logger.info("Batch job submitted successfully: %s", batch_job.name)
            return batch_job
        except Exception as e:
            logger.error("Failed to submit batch job: %s", e)
            raise

    def get_batch_job(self, job_name):
        """Retrieves the status of a batch job."""
        try:
            return self.client.batches.get(name=job_name)
        except Exception as e:
            logger.error("Failed to get batch job status for %s: %s", job_name, e)
            raise

    def retrieve_batch_results(self, job_name):
        """Retrieves the results of a completed batch job."""
        try:
            # For this SDK, usually we iterate over the results
            results = []
            logger.info("Retrieving results for batch job: %s", job_name)
            # Assuming the job is done, we traverse the results
            # Note: The actual method to get results might depend on version, 
            # but usually it's client.batches.list_results(name=...) or similar, 
            # or simply iterating if the job object has a result method.
            # However, looking at standard patterns, it might be accessing the output file.
            # Let's try the standard generator approach if available, or just log for now if unsure.
            # BUT, to be safe, let's assume we can't easily get it without proper documentation 
            # OR we try to simply use the 'output_file' property if valid.
            
            # Re-fetch job to be sure
            job = self.client.batches.get(name=job_name)
            if job.state == "ACTIVE": # processing
                 logger.info("Job is still active.")
                 return None
            
            return job 
        except Exception as e:
            logger.error("Failed to retrieve results for %s: %s", job_name, e)
            raise

    def extract_batch_results(self, batch_job):
        """
        Extracts and parses items from a completed batch job's inline responses.
        
        Args:
            batch_job: The completed batch job object
            
        Returns:
            list: Merged list of all items extracted from all responses
        """
        if not batch_job.dest or not batch_job.dest.inlined_responses:
            logger.error("No inline responses found in batch job")
            return []
        
        logger.info("Processing %d inline responses...", len(batch_job.dest.inlined_responses))
        
        raw_items = []
        for i, inline_response in enumerate(batch_job.dest.inlined_responses):
            logger.debug("Processing response %d/%d", i+1, len(batch_job.dest.inlined_responses))
            
            if inline_response.error:
                logger.error("Error in response %d: %s", i+1, inline_response.error)
                continue
                
            if not inline_response.response:
                logger.warning("No response data in response %d", i+1)
                continue
            
            # Extract text from response
            try:
                response_text = inline_response.response.text
            except AttributeError:
                logger.warning("Could not extract text from response %d", i+1)
                continue
            
            # Parse JSON from response text (same logic as transcribe_index)
            json_text = response_text
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].split("```")[0].strip()
            else:
                # Fallback: try to find the start and end of the JSON array
                start_idx = response_text.find("[")
                end_idx = response_text.rfind("]")
                if start_idx != -1 and end_idx != -1:
                    json_text = response_text[start_idx:end_idx+1].strip()
            
            try:
                data = json.loads(json_text)
                if isinstance(data, list):
                    raw_items.extend(data)
                    logger.debug("Extracted %d items from response %d", len(data), i+1)
                else:
                    logger.warning("Response %d did not contain a list", i+1)
            except json.JSONDecodeError as e:
                logger.error("Failed to parse JSON from response %d: %s", i+1, e)
                logger.debug("Raw response text: %s", response_text[:500])
                continue
        
        logger.info("Successfully retrieved %d total items from batch job", len(raw_items))
        return raw_items

    def delete_batch(self, name):
        try:
            self.client.batches.delete(name=name)
            logger.info("Batch job deleted successfully: %s", name)
        except Exception as e:
            logger.error("Failed to delete batch job %s: %s", name, e)
            raise
