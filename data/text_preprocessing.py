import re
import unicodedata


def newline_to_rune(text: str) -> str:
    """Preprocesses text by replacing various newline representations with the rune 'ᛉ' and
        additional spaces before and after the rune. Also adds the rune 'ᛉ' to the end of the text.

    Examples
    --------
    >>> newline_to_rune('max\n mustermann\n hauptstr. 123 snapaddynewline 97070 würzburg')
    'max ᛉ  mustermann ᛉ  hauptstr. 123 ᛉ 97070 würzburg ᛉ'

    Parameters
    ----------
    text : str
        String with newlines.

    Returns
    -------
    str
        String with replaced newlines.
    """
    text = text.replace("snapaddynewline", " ᛉ ")
    lines = text.splitlines()
    text = " ᛉ ".join(lines)
    return text


def trim_whitespaces(text: str) -> str:
    """Reduces all whitespace in a string to a single whitespace.

    Examples
    --------
    >>> trim_whitespaces('max  mustermann    würzburg')
    'max mustermann würzburg'

    Parameters
    ----------
    text : str
        Text with whitespaces.

    Returns
    -------
    str
        Text with trimmed whitespaces.
    """

    text = re.sub(" +", " ", text)
    text = re.sub(" {2,}", " ", text.strip())

    return text


def normalize_special_characters(text: str) -> str:
    """Normalize special characters.

    Examples
    --------
    >>> normalize_special_characters('max\xa0mustermann ’würzburg’')
    "max mustermann 'würzburg'"

    Parameters
    ----------
    text : str
        Text with special characters.

    Returns
    -------
    str
        Text without special characters.
    """

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("``", '"')
    text = text.replace("''", '"')
    text = text.replace("–", "-")
    text = text.replace("（", "(")
    text = text.replace("）", ")")

    return text


def preprocessing(text: str, lowercase: bool = True) -> str:
    """Preprocesses string. The following methods are applied:
        - Text to lowercase.
        - Replacing newlines with the rune symbol 'ᛉ' and additional spaces.
        - Normalize special characters
            - \xa0 --> ' '
            -  ’ --> '
        - Trim whitespaces to a single whitespace

    Examples
    --------
    >>> preprocessing('max\xa0\nmustermann\n97070   ’würzburg’')
    "max ᛉ mustermann ᛉ 97070 'würzburg' ᛉ"

    Parameters
    ----------
    text : str
        Unpreprocessed string.
    lowercase : bool
        Converts text to lowercase, by default True.

    Returns
    -------
    str
        Preprocessed string
    """
    if lowercase:
        text = text.lower()
    text = newline_to_rune(text)
    text = normalize_special_characters(text)
    text = trim_whitespaces(text)

    return text
