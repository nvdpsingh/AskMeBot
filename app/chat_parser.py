import re

def parse_markdown_to_html(text: str) -> str:
    """
    Parse markdown-style formatting to HTML
    """
    # Convert **bold text** to <strong>bold text</strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # Convert *italic text* to <em>italic text</em>
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    
    # Convert `code` to <code>code</code>
    text = re.sub(r'`(.*?)`', r'<code class="bg-gray-800 text-yellow-400 px-1 py-0.5 rounded text-sm">\1</code>', text)
    
    # Convert # Heading to <h1>Heading</h1>
    text = re.sub(r'^# (.*?)$', r'<h1 class="text-xl font-bold text-yellow-400 mb-2">\1</h1>', text, flags=re.MULTILINE)
    
    # Convert ## Subheading to <h2>Subheading</h2>
    text = re.sub(r'^## (.*?)$', r'<h2 class="text-lg font-bold text-yellow-300 mb-2">\1</h2>', text, flags=re.MULTILINE)
    
    # Convert ### Sub-subheading to <h3>Sub-subheading</h3>
    text = re.sub(r'^### (.*?)$', r'<h3 class="text-md font-bold text-yellow-200 mb-1">\1</h3>', text, flags=re.MULTILINE)
    
    # Convert line breaks to <br> tags
    text = text.replace('\n', '<br>')
    
    return text

def parse_llm_output(llm_output: str):
    """
    Parse LLM output and convert markdown formatting to HTML
    """
    # Parse markdown formatting to HTML
    parsed_output = parse_markdown_to_html(llm_output)
    
    return {
        "original_output": llm_output,
        "parsed_output": parsed_output,
        "formatted": True
    }