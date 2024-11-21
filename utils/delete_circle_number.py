import re
import json
import ast


def delete_circle_number(text):
    """
    Delete circle numbers (①, ②, etc.) and their corresponding Arabic numerals (1., 2., etc.)
    from problem choices in Korean text.

    Args:
        text (str): Input text containing problem choices

    Returns:
        str: Text with circle numbers removed
    """
    # Remove circle numbers (①, ②, ③, ④, ⑤)
    text = re.sub(r"[①②③④⑤]", "", text)

    # Remove Arabic numerals with dots (1., 2., 3., 4., 5.)
    text = re.sub(r"\d+\.\s*", "", text)

    return text.strip()


def process_choices(choices):
    """
    Process a list of choices to remove circle numbers.

    Args:
        choices (list): List of choice strings

    Returns:
        list: List of choices with circle numbers removed
    """
    return [delete_circle_number(choice) for choice in choices]


def process_military_korean_problems(problems_str):
    """
    Process problems from military-korean.csv format by removing circle numbers from choices.

    Args:
        problems_str (str): String containing problem dictionary in the format:
            {'question': '...', 'choices': ['①...', '②...', ...], 'answer': int}

    Returns:
        str: JSON string with processed choices
    """
    try:
        # Convert string to dictionary
        problems = ast.literal_eval(problems_str)

        # Process choices if they exist
        if "choices" in problems:
            problems["choices"] = process_choices(problems["choices"])

        # Convert back to string
        return json.dumps(problems, ensure_ascii=False)
    except:
        return problems_str
