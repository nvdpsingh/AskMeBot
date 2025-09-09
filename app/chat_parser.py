import re
import logging

# Create logger for chat parser
logger = logging.getLogger(__name__)

def parse_markdown_to_html(text: str) -> str:
    """
    Parse markdown-style formatting to HTML
    """
    logger.info("🔍 CHAT PARSER - MARKDOWN PROCESSING")
    logger.info(f"📏 Input text length: {len(text)} characters")
    
    # Convert **bold text** to <strong>bold text</strong>
    bold_count = len(re.findall(r'\*\*(.*?)\*\*', text))
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    if bold_count > 0:
        logger.info(f"✨ Converted {bold_count} bold text patterns")
    
    # Convert *italic text* to <em>italic text</em>
    italic_count = len(re.findall(r'(?<!\*)\*([^*]+)\*(?!\*)', text))
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)
    if italic_count > 0:
        logger.info(f"✨ Converted {italic_count} italic text patterns")
    
    # Convert `code` to <code>code</code>
    code_count = len(re.findall(r'`(.*?)`', text))
    text = re.sub(r'`(.*?)`', r'<code class="bg-gray-800 text-yellow-400 px-1 py-0.5 rounded text-sm">\1</code>', text)
    if code_count > 0:
        logger.info(f"✨ Converted {code_count} code patterns")
    
    # Convert # Heading to <h1>Heading</h1>
    h1_count = len(re.findall(r'^# (.*?)$', text, flags=re.MULTILINE))
    text = re.sub(r'^# (.*?)$', r'<h1 class="text-xl font-bold text-yellow-400 mb-2">\1</h1>', text, flags=re.MULTILINE)
    if h1_count > 0:
        logger.info(f"✨ Converted {h1_count} H1 headings")
    
    # Convert ## Subheading to <h2>Subheading</h2>
    h2_count = len(re.findall(r'^## (.*?)$', text, flags=re.MULTILINE))
    text = re.sub(r'^## (.*?)$', r'<h2 class="text-lg font-bold text-yellow-300 mb-2">\1</h2>', text, flags=re.MULTILINE)
    if h2_count > 0:
        logger.info(f"✨ Converted {h2_count} H2 headings")
    
    # Convert ### Sub-subheading to <h3>Sub-subheading</h3>
    h3_count = len(re.findall(r'^### (.*?)$', text, flags=re.MULTILINE))
    text = re.sub(r'^### (.*?)$', r'<h3 class="text-md font-bold text-yellow-200 mb-1">\1</h3>', text, flags=re.MULTILINE)
    if h3_count > 0:
        logger.info(f"✨ Converted {h3_count} H3 headings")
    
    # Convert line breaks to <br> tags
    line_breaks = text.count('\n')
    text = text.replace('\n', '<br>')
    if line_breaks > 0:
        logger.info(f"✨ Converted {line_breaks} line breaks to <br> tags")
    
    logger.info(f"📏 Output text length: {len(text)} characters")
    logger.info("✅ Markdown parsing completed")
    
    return text

def parse_llm_output(llm_output: str):
    """
    Parse LLM output and convert markdown formatting to HTML
    """
    logger.info("🔍 CHAT PARSER - LLM OUTPUT PROCESSING")
    logger.info(f"📏 Input length: {len(llm_output)} characters")
    
    # Parse markdown formatting to HTML
    parsed_output = parse_markdown_to_html(llm_output)
    
    logger.info("✅ LLM output parsing completed")
    logger.info("=" * 30)
    
    return {
        "original_output": llm_output,
        "parsed_output": parsed_output,
        "formatted": True
    }