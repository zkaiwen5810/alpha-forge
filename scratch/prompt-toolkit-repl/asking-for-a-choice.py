from prompt_toolkit.shortcuts import choice

result = choice(
    message="Please choose a dish:",
    options=[
        ("pizza", "Pizza with mushrooms"),
        ("salad", "Salad with tomatoes"),
        ("sushi", "Sushi"),
    ],
    default="salad",
)

print(f"Your choice: {result}")
