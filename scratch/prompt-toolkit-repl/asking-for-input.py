from prompt_toolkit import PromptSession, prompt
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style
from pygments.lexers.html import HtmlLexer

text = prompt("Give me some input: ")
print(f"You said: {text}")

# session = PromptSession()
# text1 = session.prompt()
# text2 = session.prompt()

# syntax highlighting
text = prompt("Enter HTML: ", lexer=PygmentsLexer(HtmlLexer))
print(f"You said: {text}")


# Toolbar
def bottom_toolbar():
    return [("class:bottom-toolbar", " This is a tool bar. ")]


style = Style.from_dict(
    {
        "bottom-toolbar": "#ffffff bg:#333333",
    }
)

text = prompt("> ", bottom_toolbar=bottom_toolbar, style=style)
