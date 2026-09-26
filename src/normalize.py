
# Normalization functions for text, names, and addresses
import re
import unicodedata


NAME_ABBR = {
    "pvt": "private",
    "ltd": "limited",
    "inc": "inc",
    "incorporated": "inc",
    "corp": "corporation",
    "corporation": "corporation",
    "co": "company",
    "coy": "company",

    "intl": "international",
    "natl": "national",
    "tech": "technology",
    "assoc": "associates",
    "bros": "brothers",
    "engg": "engineering",
    "grp": "group",
    "hosp": "hospital",
    "inds": "industries",
    "mgmt": "management",
    "pharma": "pharmaceuticals",
    "sys": "systems",
    "univ": "university",
    "ent": "enterprise",
    "ents": "enterprise",
    "hldgs": "holdings",
}


LEGAL_FORMS = {
    "private": "private",
    "pvt": "private",
    "limited": "limited",
    "ltd": "limited",

    "inc": "inc",
    "incorporated": "inc",

    "corporation": "corp",
    "corp": "corp",

    "company": "co",
    "co": "co",

    "llc": "llc",
    "llp": "llp",
    "lp": "lp",
    "plc": "plc",
}


ADDRESS_ABBR = {
    "street": "st",
    "str": "st",

    "road": "rd",

    "avenue": "av",
    "ave": "av",

    "boulevard": "bd",
    "blvd": "bd",

    "lane": "ln",

    "drive": "dr",
    "drv": "dr",

    "court": "ct",

    "place": "pl",

    "plaza": "plz",

    "square": "sq",

    "terrace": "ter",

    "circle": "cir",

    "highway": "hwy",

    "parkway": "pkwy",

    "expressway": "expy",

    "freeway": "fwy",

    "trail": "trl",

    "route": "rte",

    "suite": "ste",

    "apartment": "apt",
    "appt": "apt",

    "floor": "fl",

    "building": "bldg",

    "room": "rm",

    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",

    "northeast": "ne",
    "northwest": "nw",
    "southeast": "se",
    "southwest": "sw",
}


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).strip()

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = text.lower()

    text = "".join(
        char
        if unicodedata.category(char)[0] in {"L", "M", "N"}
        or char.isspace()
        else " "
        for char in text
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text

def normalize_name(name):
    """
    Normalize business name while preserving multilingual text.
    """

    text = normalize_text(name)

    if not text:
        return ""

    tokens = text.split()

    normalized_tokens = []

    for token in tokens:

        if token in NAME_ABBR:
            token = NAME_ABBR[token]

        if token in LEGAL_FORMS:
            token = LEGAL_FORMS[token]

        normalized_tokens.append(token)

    return " ".join(normalized_tokens)

def normalize_name_core(name):
    """
    Remove common legal/company words so that the
    meaningful business name remains.
    """

    text = normalize_name(name)

    if not text:
        return ""

    legal_words = {
        "private",
        "limited",
        "inc",
        "corp",
        "co",
        "llc",
        "llp",
        "lp",
        "plc",
        "company",
        "corporation",
    }

    tokens = [
        token
        for token in text.split()
        if token not in legal_words
    ]

    return " ".join(tokens)

def normalize_address(address):
    """
    Normalize addresses while preserving multilingual text.
    """

    text = normalize_text(address)

    if not text:
        return ""

    tokens = text.split()

    normalized_tokens = []

    for token in tokens:

        if token in ADDRESS_ABBR:
            token = ADDRESS_ABBR[token]

        normalized_tokens.append(token)

    return " ".join(normalized_tokens)

def get_address_tokens(address):
    """
    Return useful address tokens for later blocking.
    """

    text = normalize_address(address)

    if not text:
        return []

    return text.split()

def get_address_numbers(address):
    """
    Extract numbers from an address.

    Useful later for matching house numbers,
    postal codes, unit numbers, etc.
    """

    text = normalize_address(address)

    if not text:
        return []

    return re.findall(r"\d+", text)


def normalize_country(country):
    """
    Keep country handling simple and open-ended.
    We do not hardcode only US/India.
    """

    if country is None:
        return ""

    return normalize_text(country)




