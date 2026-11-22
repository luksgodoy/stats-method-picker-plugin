"""Structured data extraction from HTML pages (parsing, not network requests -
the capture itself depends on urllib/requests, out of the testable scope here)."""
from html.parser import HTMLParser


class _TableExtractor(HTMLParser):
    """Extracts text from specific tags (e.g. table rows, headings) of an HTML document
    already fetched - separates the parsing step (testable offline) from the network step."""

    def __init__(self, target_tags):
        super().__init__()
        self.target_tags = set(target_tags)
        self._stack = []
        self.records = []
        self._current = None

    def handle_starttag(self, tag, attrs):
        self._stack.append(tag)
        if tag in self.target_tags:
            self._current = []

    def handle_endtag(self, tag):
        if tag in self.target_tags and self._current is not None:
            self.records.append(" ".join(self._current).strip())
            self._current = None
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()

    def handle_data(self, data):
        text = data.strip()
        if text and self._current is not None:
            self._current.append(text)


def extract_by_tag(html, target_tags=("td", "h1", "h2", "li")):
    parser = _TableExtractor(target_tags)
    parser.feed(html)
    return {"n_records": len(parser.records), "records": parser.records}


if __name__ == "__main__":
    sample_html = """
    <html><body>
      <h1>Product Catalog</h1>
      <ul>
        <li>Product A - $10</li>
        <li>Product B - $20</li>
      </ul>
      <table>
        <tr><td>Product A</td><td>10</td></tr>
        <tr><td>Product B</td><td>20</td></tr>
      </table>
    </body></html>
    """
    r = extract_by_tag(sample_html)
    print("records extracted:", r["n_records"])
    print(r["records"])
