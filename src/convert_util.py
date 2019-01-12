
from hanziconv import HanziConv
from corpus_cleaner.util import logger_util

logger = logger_util.get_logger()

def convert_simplified_to_traditional(line:str):
    """Convert simplified chinese to simplified chinese
    """
    if not line:
        logger.error("the line is null...")
        return
    
    return HanziConv.toTraditional(line)



def convert_traditional_to_simplified(line:str):
    """Convert traditional chinese to simplified chinese
    """
    if not line:
        logger.error("the line is null...")
        return

    return HanziConv.toSimplified(line)


def convert_digits_to_chinese(line:str):
"""
    convert digits to chinese word. And the steps is below:
    1, whether the digits belong to phoneNumber
    2, whether the digits belong to years
    3, whether the digits belong to special groups
    4, whether the digits belong to real digits
"""    
    if not line:
        return
