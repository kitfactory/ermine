import logging

import logging

logger = logging.getLogger("logger")    #logger名loggerを取得
logger.setLevel(logging.DEBUG)  #loggerとしてはDEBUGで

#handler2を作成
handler = logging.FileHandler(filename="logtest.log")  #handler2はファイル出力
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
logger.addHandler(handler)



obj = {"hoge":"hoge"}

logger.debug("hello",obj)
logger.debug(None)
