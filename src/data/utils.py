import os.path as op
import urllib.request

root_path = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))


def request_raw_data_by_url(url: str, raw_data_path: str):
    data_path = op.join(root_path, raw_data_path)
    urllib.request.urlretrieve(url, data_path)
