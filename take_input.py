from vectorise_medium import search_query_fast
import logging
from logging.handlers import RotatingFileHandler

log_name = 'take_input.py.log'
logging.basicConfig(filename=log_name, format='%(levelname)s:%(message)s', level=logging.DEBUG)
log = logging.getLogger()
handler = RotatingFileHandler(log_name, maxBytes=1048576)
log.addHandler(handler)

def take_input(encoded_query, n_best=300, threshold=10):
    # take query and find results
    query_result = search_query_fast(encoded_query=encoded_query,
                                     n_best=n_best,
                                     threshold=threshold)
    # log.info(query_result)

    return query_result
