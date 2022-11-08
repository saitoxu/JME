import os
import logging


def getLogger(name, path, level=logging.DEBUG):
    # ロガーの定義
    logger = logging.getLogger(name)
    logger.setLevel(level)
    os.makedirs(path, exist_ok=True)
    # フォーマットの定義
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s: %(message)s")
    log_file = f'{path}train.log'
    # ログファイルの中身をクリアする
    with open(log_file, 'w'):
        pass
    # ファイル書き込み用
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    # コンソール出力用
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    # それぞれロガーに追加
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger
