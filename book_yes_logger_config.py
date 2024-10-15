# logger_config.py
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 配置并发安全的日志处理器
    handler = ConcurrentRotatingFileHandler("/nvme0n1-disk/book_yes/logs/book_yes_app.log", maxBytes=5 * 1024 * 1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = setup_logger()