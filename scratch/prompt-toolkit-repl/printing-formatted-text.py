from prompt_toolkit import HTML, print_formatted_text
from prompt_toolkit.formatted_text import FormattedText, to_formatted_text
from prompt_toolkit.styles import Style

# Colors from the ANSI palette.
print_formatted_text(HTML("<ansired>This is red</ansired>"))
# Named colors (256 color palette, or true color, depending on the output).
print_formatted_text(HTML("<violet>This is violet</violet>"))
# Both foreground and background colors can also be specified setting the fg and bg attributes of any HTML tag:
print_formatted_text(HTML('<aaa fg="ansiwhite" bg="ansigreen">White on green</aaa>'))

# Underneath, all HTML tags are mapped to classes from a stylesheet, so you can assign a style for a custom tag.
style = Style.from_dict(
    {
        "aaa": "#ff0066",
        "bbb": "#44ff00 italic",
    }
)
print_formatted_text(HTML("<aaa>Hello</aaa> <bbb>world</bbb>!"), style=style)

# Internally, both HTML and ANSI objects are mapped to a list of (style, text) tuples. It is however also possible to create such a list manually with FormattedText class. This is a little more verbose, but it’s probably the most powerful way of expressing formatted text.
text = FormattedText(
    [
        ("#ff0066", "Hello"),
        ("", " "),
        ("#44ff00 italic", "World"),
    ]
)
print_formatted_text(text)

# Similar to the HTML example, it is also possible to use class names, and separate the styling in a style sheet.
# The text.
text = FormattedText(
    [
        ("class:aaa", "Hello"),
        ("", " "),
        ("class:bbb", "World"),
    ]
)

# The style sheet.
style = Style.from_dict(
    {
        "aaa": "#ff0066",
        "bbb": "#44ff00 italic",
    }
)

print_formatted_text(text, style=style)

html = HTML("<aaa>Hello</aaa> <bbb>world</bbb>!")
text = to_formatted_text(html, style="class:my_html bg:#00ff00 italic")
print_formatted_text(text)
