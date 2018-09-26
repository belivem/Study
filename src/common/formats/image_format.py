import tensorflow as tf

import logging

def get_defaultLog(func_name,file_level,std_level):

    #Instructions
    #logger.debug("It is a debug...")
    #logger.info("It is a info, and name is %","lee")
    #logger.warn("It is a warning...")
    #logger.error("It is a error...")
     
    #must set default level of logger, and if handler's level < default level, the part of low level not effective
    logger_name = "lee"
    logger = logging.getLogger(logger_name)
    logger.setLevel(file_level)         

    #set FileHandler
    log_path = "D:\\vsCode_workSpace\\other_data\\log\\%s.log"%(func_name)
    fh = logging.FileHandler(log_path)
    fh.setLevel(file_level)

    #set StreamHandler
    sh = logging.StreamHandler(stream=None)
    sh.setLevel(std_level)

    fmt = "%(asctime)s - %(levelname)s - %(name)s - %(filename)s: %(message)s"
    datefmt = "%m/%d/%Y %H:%M:%S %p"
    formatter = logging.Formatter(fmt,datefmt)
    
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger
