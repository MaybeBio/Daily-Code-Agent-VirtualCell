"""Download pinned public dataset assets, retaining resumable partial files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import requests
from bio.io.paths import PROJECT_ROOT


def ranged_download(item, partial, connections=8):
    """Bound memory and preserve a contiguous resumable prefix while parallelizing."""
    offset = partial.stat().st_size if partial.exists() else 0
    total = item['size']
    if offset == total: return
    if offset > total: raise ValueError('Partial exceeds expected file size')
    chunk_size = 64*1024*1024
    folder = partial.with_suffix('.chunks')
    folder.mkdir(exist_ok=True)
    def fetch(start):
        end = min(total-1,start+chunk_size-1)
        path = folder/f'{start}-{end}'
        if path.exists() and path.stat().st_size == end-start+1: return path
        last_error = None
        for attempt in range(4):
            try:
                with requests.get(item['url'], params={'cellbench_range': str(start)}, headers={'Range':f'bytes={start}-{end}', 'Accept-Encoding':'identity'},
                                  stream=True,timeout=(45,120)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
                        raise ValueError('Server did not honor requested byte range')
                    with path.open('wb') as handle:
                        for block in response.iter_content(4*1024*1024):handle.write(block)
                if path.stat().st_size != end-start+1:raise ValueError('Incomplete chunk')
                return path
            except Exception as exc:
                last_error=exc
                print(f"CHUNK RETRY {start} {attempt+1}: {exc}",flush=True)
                time.sleep(2)
        raise last_error
    starts=list(range(offset,total,chunk_size))
    ready={};failures=[];last=time.monotonic()
    with ThreadPoolExecutor(max_workers=connections) as executor:
        futures={executor.submit(fetch,start):start for start in starts}
        for future in as_completed(futures):
            try:ready[futures[future]]=future.result()
            except Exception as exc:failures.append(str(exc))
            if time.monotonic()-last>30:
                print(f"CHUNKS {item['id']} {len(ready)}/{len(starts)} ready",flush=True);last=time.monotonic()
    if failures:raise ValueError(f'{len(failures)} failed chunks: {failures[0]}')
    with partial.open('ab') as handle:
        for start in starts:
            path=ready[start]
            with path.open('rb') as chunk:
                for block in iter(lambda:chunk.read(8*1024*1024),b''):handle.write(block)
            path.unlink()
            if time.monotonic()-last>30:
                print(f"MERGE {item['id']} {handle.tell()/1e9:.2f} GB",flush=True);last=time.monotonic()
    folder.rmdir()



def fetch_binary(url, dest, expected=None):
    import subprocess
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'START {dest.name}', flush=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    cmd = ['curl', '-L', '--fail', '--retry', '4', '--retry-delay', '2',
           '--connect-timeout', '30', '-o', str(tmp), url]
    subprocess.run(cmd, check=True)
    if expected and tmp.stat().st_size != expected:
        raise ValueError('File size mismatch')
    tmp.replace(dest)
    print(f'FETCHED {dest.name} {dest.stat().st_size / 1e6:.1f} MB', flush=True)
    return dest


def read_10x_archive(archive):
    import tarfile, tempfile
    from scanpy import read_10x_mtx
    tmp = tempfile.TemporaryDirectory()
    with tarfile.open(archive) as tar:
        tar.extractall(tmp.name)
    matrices = list(Path(tmp.name).rglob('matrix.mtx')) + list(Path(tmp.name).rglob('matrix.mtx.gz'))
    if not matrices:
        raise ValueError(f'No 10x matrix in {archive}')
    return read_10x_mtx(str(matrices[0].parent), var_names='gene_symbols', cache=False), tmp


def assemble_10x(item):
    import anndata as ad
    import pandas as pd
    folder = (PROJECT_ROOT / item['path']).parent
    archives = [(folder / f"{item['id']}_0.tar.gz", item.get('batch'))]
    fetch_binary(item['url'], archives[0][0], item.get('size'))
    for i, part in enumerate(item.get('parts') or [], start=1):
        path = folder / f"{item['id']}_{i}.tar.gz"
        fetch_binary(part['url'], path, part.get('size'))
        archives.append((path, part.get('batch')))
    pieces, temps = [], []
    for archive, batch in archives:
        adata, tmp = read_10x_archive(archive)
        temps.append(tmp)
        if batch:
            adata.obs['batch'] = batch
        pieces.append(adata)
    if len(pieces) == 1:
        data = pieces[0]
    else:
        keys = [a.obs['batch'].iloc[0] if 'batch' in a.obs else str(i) for i, a in enumerate(pieces)]
        data = ad.concat(pieces, join='outer', keys=keys, index_unique='-')
    if item.get('annotation_url'):
        annot_path = folder / f"{item['id']}_annotation.tsv"
        fetch_binary(item['annotation_url'], annot_path)
        table = pd.read_csv(annot_path, sep='\t')
        key = 'barcodes' if 'barcodes' in table.columns else table.columns[0]
        table = table.drop_duplicates(key).set_index(key)
        matched = table.reindex(data.obs_names.astype(str))
        if 'celltype' not in table.columns:
            raise ValueError('Annotation table has no celltype column')
        if matched['celltype'].notna().mean() < 0.9:
            stripped = data.obs_names.astype(str).str.replace(r'-\d+$', '', regex=True)
            matched = table.reindex(stripped)
        data.obs['celltype'] = matched['celltype'].astype(str).to_numpy()
        missing = data.obs['celltype'].isin(['nan', 'None', 'NaN']).mean()
        if missing > 0.5:
            raise ValueError('Failed to attach 10x cell-type labels')
    if 'batch' not in data.obs:
        data.obs['batch'] = item.get('batch', item['id'])
    data.obs['donor_id'] = item.get('donor', 'donor_a')
    if 'celltype' not in data.obs:
        data.obs['celltype'] = 'PBMC'
    target = PROJECT_ROOT / item['path']
    data.write_h5ad(target, compression='lzf')
    for tmp in temps:
        tmp.cleanup()
    return target


def download_asset(item):
    target = PROJECT_ROOT / item['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = target.with_suffix('.download.json')
    if target.is_file() and receipt.is_file():
        record = json.loads(receipt.read_text())
        if target.stat().st_size == record['size'] and record['url'] == item['url']:
            return {'id': item['id'], 'status': 'existing', **record}
    if item.get('layout') == '10x_mtx':
        assembled = assemble_10x(item)
        hashes = {k: hashlib.new(k) for k in ('sha256', 'md5')}
        with assembled.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                for digest in hashes.values():
                    digest.update(block)
        record = dict(url=item['url'], size=assembled.stat().st_size,
                      sha256=hashes['sha256'].hexdigest(), md5=hashes['md5'].hexdigest(),
                      layout='10x_mtx')
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        print(f"DONE {item['id']} {record['size']/1e9:.2f} GB", flush=True)
        return dict(id=item['id'], status='downloaded', **record)
    partial = target.with_suffix(target.suffix+'.part')
    for attempt in range(4):
        try:
            if item.get('size',0)>8_000_000_000:
                ranged_download(item,partial)
            else:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {'Accept-Encoding':'identity'}
                if offset: headers['Range'] = f'bytes={offset}-'
                print(f"START {item['id']} offset={offset}", flush=True)
                with requests.get(item['url'], headers=headers, stream=True, timeout=(45,120)) as response:
                    response.raise_for_status()
                    if response.status_code == 206 and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected Content-Range')
                    append = offset > 0 and response.status_code == 206
                    received = 0
                    expected = response.headers.get('Content-Length')
                    last = time.monotonic()
                    with partial.open('ab' if append else 'wb') as handle:
                        for block in response.iter_content(8*1024*1024):
                            handle.write(block); received += len(block)
                            if time.monotonic()-last > 30:
                                print(f"PROGRESS {item['id']} {handle.tell()/1e9:.2f} GB",flush=True);last=time.monotonic()
                    if expected and received != int(expected): raise ValueError('Incomplete response')
            if item.get('size') and partial.stat().st_size != item['size']: raise ValueError('File size mismatch')
            hashes = {k:hashlib.new(k) for k in ('sha256','md5')}
            with partial.open('rb') as handle:
                magic = handle.read(8)
                if item.get('compression') == 'gzip':
                    if not magic.startswith(b'\x1f\x8b'): raise ValueError('Not a gzip file')
                elif magic != b'\x89HDF\r\n\x1a\n': raise ValueError('Not an HDF5 file')
                handle.seek(0)
                for block in iter(lambda: handle.read(8*1024*1024), b''):
                    for digest in hashes.values(): digest.update(block)
            if item.get('md5') and hashes['md5'].hexdigest() != item['md5']:
                partial.unlink(); raise ValueError('Official MD5 mismatch')
            if item.get('compression') != 'gzip':
                import h5py
                with h5py.File(partial,'r') as f:
                    if not all(k in f for k in ('X','obs','var')): raise ValueError('Missing AnnData components')
            partial.replace(target)
            record = dict(url=item['url'],size=target.stat().st_size,sha256=hashes['sha256'].hexdigest(),md5=hashes['md5'].hexdigest())
            receipt.write_text(json.dumps(record,indent=2)+'\n')
            print(f"DONE {item['id']} {record['size']/1e9:.2f} GB",flush=True)
            return dict(id=item['id'],status='downloaded',**record)
        except Exception as exc:
            print(f"RETRY {item['id']} {attempt+1}: {exc}",flush=True)
            if attempt==3:return dict(id=item['id'],status='failed',error=str(exc))
            time.sleep(2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ids',nargs='+');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    assets=json.loads((PROJECT_ROOT/'resources/dataset_assets.json').read_text())
    if args.ids:
        missing=set(args.ids)-{a['id'] for a in assets}
        if missing:p.error(f'Unknown datasets: {sorted(missing)}')
        assets=[a for a in assets if a['id'] in args.ids]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results=list(executor.map(download_asset,assets))
    suffix = '_'.join(args.ids) if args.ids else 'all'
    path=PROJECT_ROOT/f'output/dataset_downloads_{suffix}.json';path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(results,indent=2)+'\n')
    return int(any(r['status']=='failed' for r in results))


if __name__=='__main__':raise SystemExit(main())
