# Index Linking

This project extracts index items from PDF files using Google Gemini, links them to Archive.org pages, and generates an HTML report. 

**Note: As PoC, currently only the news paper index of [Mennonitische Rundschau](https://archive.org/details/mr-index-1910-1919-web/) is supported.**

## Features

- **Parallel Processing**: Process PDF pages concurrently for faster extraction (configurable with `--workers`)
- **Batch Mode**: Submit jobs to Gemini Batch API for cost-effective processing of large documents
- **Flexible Models**: Support for multiple Gemini models including `gemini-2.5-flash`, `gemini-3-flash-preview`, and `gemini-3-pro-preview`
- **Customizable Prompts**: Use custom prompt files to tailor extraction behavior
- **Link to full texts**: Automatically link extracted items to Archive.org full texts

## Project Structure

```
mr_linkmyindex/
├── src/                      # Source code
│   ├── linkmyindex.py      # Main entry point
│   ├── gemini_service.py     # Gemini API integration
│   ├── pdf_processor.py      # PDF processing utilities
│   └── template_service.py   # HTML template rendering
├── prompts/                  # Prompt templates
├── templates/                # Jinja2 HTML templates
├── data/                     # Input data and mappings
│   └── itemlist.txt          # Date-to-Archive.org ID mappings (needed for MR index)
├── generated/                # Output directory (gitignored)
├── logs/                     # Log files (gitignored)
└── price_table.json          # Gemini API pricing configuration (for rough pricing calculation)
```

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Gemini API key (set as `GEMINI_API_KEY` environment variable)

### Using uv (Recommended)

```bash
# Install dependencies
uv sync

# Set your API key
export GEMINI_API_KEY="your-api-key-here"
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt

# Set your API key
export GEMINI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage (Parallel Processing)

Process a PDF with 10 parallel workers (default):

**Using the shell script (recommended):**
```bash
./linkmyindex.sh -i data/your_index.pdf -o ./generated
```

**Direct Python call:**
```bash
python src/linkmyindex.py -i data/your_index.pdf -o ./generated
```

### Batch Mode (Cost-Effective for Large Documents)

Submit a batch job to Gemini:

```bash
./linkmyindex.sh -i data/your_index.pdf --batch-mode
```

Check batch job status (run the same command again):

```bash
./linkmyindex.sh -i data/your_index.pdf --batch-mode
```

When the job completes, results are automatically retrieved and processed.

### Advanced Options

```bash
./linkmyindex.sh \
  -i data/MR_Index_1910-1919.pdf \
  -o ./generated \
  -m gemini-2.5-flash \
  -w 20 \
  --thought-level medium \
  --prompt-file prompts/custom.txt \
  --keep-temporary-files
```

## Command-Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--index-file` | `-i` | **required** | Input PDF index file |
| `--itemlist` | `-l` | `data/itemlist.txt` | Path to date-to-ID mapping file |
| `--template` | `-t` | `index.html.j2` | HTML template name |
| `--prompt-file` | `-p` | `prompts/mr.txt` | Prompt file for Gemini |
| `--model` | `-m` | `gemini-3-flash-preview` | Gemini model to use |
| `--thought-level` | | `low` | Thinking level (`low`, `medium`, `high`, `minimal`) |
| `--output-dir` | `-o` | `generated` | Output directory |
| `--json-file` | `-j` | _auto_ | JSON output filename (auto-derived from input) |
| `--force-ocr` | `-f` | _false_ | Force re-extraction even if JSON exists |
| `--verbose` | `-v` | _false_ | Enable debug logging |
| `--skip-index-creation` | `-si` | _false_ | Only extract JSON, skip HTML generation |
| `--workers` | `-w` | `10` | Number of parallel workers (1 = sequential) |
| `--keep-temporary-files` | `-k` | _false_ | Keep split PDF pages after processing |
| `--batch-mode` | `-b` | _false_ | Use Gemini Batch API |

## Output Files

### Parallel Mode
- `generated/<input_name>.json`: Extracted index items in JSON format
- `generated/index.html`: Linked HTML index report
- `logs/linkmyindex.log`: Detailed processing logs

### Batch Mode
- `generated/batch.json`: Batch job tracking metadata (auto-deleted on completion)
- `generated/<input_name>.json`: Extracted items (created when job completes)
- `generated/index.html`: HTML report (created when job completes)

## Configuration

### Pricing Table (`price_table.json`)

Configure Gemini API pricing and USD→EUR exchange rate:

```json
{
  "usd_to_eur": 0.96,
  "pricing": {
    "gemini-2.5-flash": {
      "input": 0.075,
      "output": 0.30
    },
    "gemini-3-flash-preview": {
      "input": 0.0375,
      "output": 0.15
    }
  }
}
```

### Custom Prompts

Create custom prompt files in `prompts/` and specify with `--prompt-file`:

```bash
python src/linkmyindex.py -i data/input.pdf -p prompts/custom.txt
```

## Development

### Running Tests

```bash
python -m pytest
```

### Code Style

The project uses Pylint for linting. Run checks with:

```bash
pylint src/
```

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
