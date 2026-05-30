from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

_TPL_DIR = Path(__file__).parent / "templates"


def render_report(results: dict, out_path) -> None:
    env = Environment(
        loader=FileSystemLoader(_TPL_DIR),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["money"] = lambda v: f"{v:,.0f}"
    html = env.get_template("report.html.j2").render(**results)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html)
