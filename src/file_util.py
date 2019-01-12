import os
import time
import platform
import codecs

from corpus_cleaner.util import logger_util

logger = logger_util.get_logger()


def file_existed(file_path):
    """Judge the file existed or not"""

    if os.path.exists(file_path) and os.path.isfile(file_path):
        return True
    else:
        return False


def dir_existed(dir_path):
    """Judge the dir existed or not"""

    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        return True
    else:
        return False


def dir_existed_and_created(dir_path):
    """Judge the dir existed or not, 
        if not, created it,
        if existed, empty it
    """
    if not dir_existed(dir_path):
        os.makedirs(dir_path)
        logger.warn("dir {} not existed, but has created successful...".format(
            dir_path))
    else:
        for file_name in os.listdir(dir_path):
            os.remove(os.path.join(dir_path, file_name))


def split_big_file(big_file_path, dest_dir, split_prefix, number):
    """split big file into number smaller files by linux_command,
        and return path list of smaller files
    """

    start = time.time()
    if "Windows" in platform.system():
        msg = "method of split just suitable for linux"
        raise EnvironmentError(msg)

    path_list = []
    # 1, get lines_count from big file
    result = os.popen("wc -l {}".format(path_from))
    res = result.read()
    lines_count = int(res.split()[0])

    # 2, split source file to number files
    lines_count_one_file = int(lines_count // number) + 1
    path_to_prefix = os.path.join(path_to_prefix, asplit_prefix)
    ret_code = os.system("split -l %d %s %s" % (
        lines_count_one_file, path_from, path_to_prefix))
    if ret_code != 0:
        logger.info("split big file is failed, please recheck it...")
        return path_list

    # 3, return files_path
    path_list = [os.path.join(dest_dir, file_name)
                 for file_name in os.listdir(dest_dir)]

    end = time.time()
    print("split file by lines cost %ds, and generate %d new files..." % (
        (end - start), number))
    return path_list


def sort_file(file_path:str, sorted_by_column:int, spearator:str):
    """"sorted file by specify column.
        1, file is existed or not
        2, sorted file by specify column
    """
    start = time.time()
    if not file_existed(file_path):
        logger.error("file is not existed, please check it, and file_path is %s", file_path)
        return
    
    file_dir = os.path.dirname(file_path)
    sorted_file_name = os.path.basename(file_path) + ".sorted"
    sorted_file_path = os.path.join(file_dir, sorted_file_name)
    
    dict_file = {}
    with codecs.open(filename=file_path, mode="r", encoding="utf-8") as f_reader:
        for line in f_reader:
            tokens = line.split(spearator)
            if len(tokens) < sorted_by_column + 1:
                logger.error("sorte by column is out of bounds, size of line is %d", len(tokens))
                raise EnvironmentError()
            specify_column = tokens[sorted_by_column]
            if specify_column in dict_file.keys():
                logger.error("key is existed already, and key is %s", specify_column)
                return
            else:
                dict_file[specify_column] = line.strip()
    
    sorted_key_list = sorted(dict_file.keys())
    if not sorted_key_list:
        logger.error("sorted list is empty, please check it...")
        return
    
    with codecs.open(filename = sorted_file_path, mode="w", encoding="utf-8") as f_writer:
        for key in sorted_key_list:
            sorted_line = dict_file[key]
            f_writer.write("{}\n".format(sorted_line))

    end = time.time()
    logger.info("soeted file %s is completed, and costs %ds...", file_path, (end-start))
