import os
import logging
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

class TemplateService:
    def __init__(self, template_dir="templates"):
        self.template_dir = template_dir
        if not os.path.exists(self.template_dir):
            logger.critical("Template directory not found: %s", self.template_dir)
            raise FileNotFoundError(f"Template directory not found: {self.template_dir}")

        self.env = Environment(loader=FileSystemLoader(self.template_dir))

    def render_index(self, items, template_name="index.html.j2"):
        """Groups items by topic and renders the HTML template."""
        grouped = {}
        for itm in items:
            tpc = itm.get('topic', 'General Index')
            if tpc not in grouped:
                grouped[tpc] = []
            grouped[tpc].append(itm)
        
        try:
            tpl = self.env.get_template(template_name)
            return tpl.render(grouped=grouped)
        except Exception as e:
            logger.error("Error rendering template '%s': %s", template_name, e)
            raise

    def save_report(self, html_content, output_path):
        """Saves the rendered HTML content to a file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f_out:
                f_out.write(html_content)
            logger.info("Report saved to: %s", output_path)
        except Exception as e:
            logger.error("Failed to save report to '%s': %s", output_path, e)
            raise
