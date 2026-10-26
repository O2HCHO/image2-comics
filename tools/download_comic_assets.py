"""Build-time pinned downloads. Range chunks are checked before atomic assembly."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import time
import requests

ROOT = Path(__file__).resolve().parent.parent
CTD_REV = '440b978563c71b758e31aaa315d100faba1efa2f'
BUBBLE_REV = '1035c7ffcb91ce12f484ace1b541471575911cae'


def download(url, destination, size, headers=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == size:
        return
    response = requests.get(url, headers={**(headers or {}),'Range':'bytes=0-0'}, allow_redirects=False, timeout=40, stream=True)
    if response.is_redirect:
        target = response.headers['Location']
    else:
        response.raise_for_status()
        target = url
    response.close()
    folder = ROOT / 'build/model-prep/chunks' / destination.name
    folder.mkdir(parents=True, exist_ok=True)
    step = 512 * 1024
    def chunk(start):
        end = min(size, start+step)-1
        path = folder / str(start)
        if path.exists() and path.stat().st_size == end-start+1:
            return path
        for attempt in range(6):
            try:
                r = requests.get(target, headers={**(headers or {}),'Range': f'bytes={start}-{end}'}, timeout=(20,90))
                r.raise_for_status()
                if r.status_code != 206 or len(r.content) != end-start+1 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                    raise ValueError('Range response mismatch')
                path.write_bytes(r.content)
                return path
            except (requests.RequestException, ValueError):
                if attempt == 5:
                    raise
                time.sleep(attempt+1)
    with ThreadPoolExecutor(max_workers=16) as pool:
        paths = list(pool.map(chunk, range(0,size,step)))
    temporary = destination.with_suffix('.download')
    with temporary.open('wb') as stream:
        for path in paths:
            stream.write(path.read_bytes())
    if temporary.stat().st_size != size:
        raise ValueError('Incomplete model')
    temporary.replace(destination)


def main():
    download('https://api.github.com/repos/zyddnys/manga-image-translator/releases/assets/55024546', ROOT/'assets/models/comictextdetector.onnx',94669756,{'Accept':'application/octet-stream'})
    print('CTD complete',flush=True)
    # GitHub media endpoint serves the pinned Git blob without the raw host.
    download(f'https://api.github.com/repos/ebhon/YOLO-manga-bubble-detector/contents/best.pt?ref={BUBBLE_REV}', ROOT/'build/model-prep/bubble.pt',6217891,{'Accept':'application/vnd.github.raw+json'})
    print('Bubble complete',flush=True)
    for repo,rev,name in [('dmMaze/comic-text-detector',CTD_REV,'CTD-GPL-3.0.txt'),('ebhon/YOLO-manga-bubble-detector',BUBBLE_REV,'Bubble-MIT.txt')]:
        r = requests.get(f'https://api.github.com/repos/{repo}/contents/LICENSE?ref={rev}',headers={'Accept':'application/vnd.github.raw+json'},timeout=40)
        r.raise_for_status()
        (ROOT/'licenses'/name).write_bytes(r.content)
    print('Licenses complete',flush=True)


if __name__ == '__main__':
    main()
