import re

from corpus_cleaner.util import logger_util
logger = logger_util.get_logger()

chinese_pattern = re.compile(u'[\u4e00-\u9fa5]+')
japanese_pattern_katakana = re.compile(u'[\u30a0-\u30ff]+')
japanese_pattern_hiragana = re.compile(u'[\u3040-\u309f]+')
korean_pattern = re.compile(u'[\uac00-\ud7ff]+')
english_pattern = re.compile(u'[a-zA-Z]+')
digits_pattern = re.compile(u'[0-9]+')


def contain_chinese(line: str):
    """Contain chinese character in line or not
    """
    if not line:
        logger.error("the line is null...")
        return False

    match_obj = re.search(chinese_pattern, line)
    if match_obj:
        return True
    else:
        return False


def contain_digits(line: str):
    """Contain english character in line or not
    """
    if not line:
        logger.error("the line is null...")
        return False

    match_obj = re.search(english_pattern, line)
    if match_obj:
        return True
    else:
        return False


def contain_english(line: str):
    """Contain digits character in line or not
    """
    if not line:
        logger.error("the line is null...")
        return False

    match_obj = re.search(digits_pattern, line)
    if match_obj:
        return True
    else:
        return False


def contain_japanese(line: str):
    """Contain japanese character in line or not
    """
    if not line:
        logger.error("the line is null...")
        return False

    match_obj = re.search(japanese_pattern_hiragana, line)
    if match_obj:
        return True
    else:
        match_obj = re.search(japanese_pattern_katakana, line)
        if match_obj:
            return True
        else:
            return False


def contain_korean(line:str):
    """Contain korean character in line or not
    """
    if not line:
        logger.error("the line is null...")
        return False
    
    match_obj = re.search(korean_pattern, line)
    if match_obj:
        return True
    else:
        return False
