
import time, sys
import sys
import argparse
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
weight_id = 449759
CHUNK_SIZE = 1638400
TOKEN_FILE = Path.home() / '.civitai' / 'config'
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

def download_file_c(lora_id: str, output_path="/nvme0n1-disk/civitai-downloader/", token = "6342a8596b95d4899ca749fb8f044928"):
    headers = {
        'Authorization': f'Bearer {token}',
        'User-Agent': USER_AGENT,
    }
    url = f"https://civitai.com/api/download/models/{lora_id}"
    class NoRedirection(urllib.request.HTTPErrorProcessor):
        def http_response(self, request, response):
            return response

        https_response = http_response

    request = urllib.request.Request(url, headers=headers)
    opener = urllib.request.build_opener(NoRedirection)
    response = opener.open(request)

    if response.status in [301, 302, 303, 307, 308]:
        redirect_url = response.getheader('Location')

        parsed_url = urlparse(redirect_url)
        query_params = parse_qs(parsed_url.query)
        content_disposition = query_params.get('response-content-disposition', [None])[0]

        if content_disposition:
            filename = unquote(content_disposition.split('filename=')[1].strip('"'))
        else:
            raise Exception('Unable to determine filename')

        response = urllib.request.urlopen(redirect_url)
    elif response.status == 404:
        raise Exception('File not found')
    else:
        raise Exception('No redirect found, something went wrong')

    total_size = response.getheader('Content-Length')

    if total_size is not None:
        total_size = int(total_size)
    print('check down')
    return response, output_path, filename, total_size

# def main():
#     global args
#     args = get_args()
#
#     try:
#         download_file(args.url)
#     except Exception as e:
#         print(f'ERROR: {e}')
#
# if __name__ == '__main__':
#     main()
response, output_path, filename, total_size = download_file_c(weight_id)
CHUNK_SIZE = 1638400
output_file = filename
with open(output_file, 'wb') as f:
    downloaded = 0
    start_time = time.time()
    print(f'start do this {start_time}')
    while True:
        chunk_start_time = time.time()
        buffer = response.read(CHUNK_SIZE)
        chunk_end_time = time.time()
        if not buffer:
            break
        downloaded += len(buffer)
        f.write(buffer)
        chunk_time = chunk_end_time - chunk_start_time

        if chunk_time > 0:
            speed = len(buffer) / chunk_time / (1024 ** 2)
        if total_size is not None:
            progress = downloaded / total_size
            sys.stdout.write(f'\rDownloading: {filename} [{progress * 100:.2f}%] - {speed:.2f} MB/s')
            sys.stdout.flush()