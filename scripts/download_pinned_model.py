"""Stream pinned public model assets on the server and verify original content."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from skillcomp.runtime import json_write,status


def valid(path,entry):
    if not path.is_file() or path.stat().st_size!=entry['size']:return False
    if entry.get('lfs'):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()==entry['lfs']['sha256']
    data=path.read_bytes();return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==entry['blobId']


def download(config,out):
    target=Path(config['target_root'])/(config['model'].split('/')[-1]+'_'+config['revision']);target.mkdir(exist_ok=True);records=[]
    for entry in config['files']:
        name=entry['rfilename'];assert '/' not in name and '\\' not in name and name not in ['.','..'];path=target/name
        if path.exists():
            if not valid(path,entry):raise RuntimeError('Existing asset differs from pinned official source: '+name)
            records.append({'file':name,'bytes':entry['size'],'already_present':True});continue
        part=path.with_name(path.name+'.part')
        if valid(part,entry):part.rename(path);records.append({'file':name,'bytes':entry['size'],'resumed_complete':True});continue
        errors=[]
        for template in config['transport_url_templates']:
            try:
                if part.exists() and part.stat().st_size>=entry['size']:part.rename(part.with_name(part.name+f'.invalid_{int(time.time())}'))
                offset=part.stat().st_size if part.exists() else 0;headers={'User-Agent':'skillcomp-pinned-assets/1.0'}
                if offset:headers['Range']=f'bytes={offset}-'
                url=template.format(name=name)
                with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as response:
                    resume=offset>0 and response.status==206
                    if resume and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise RuntimeError('Download range mismatch')
                    written=offset if resume else 0;last=written
                    with part.open('ab' if resume else 'wb') as stream:
                        while chunk:=response.read(4*1024*1024):
                            stream.write(chunk);written+=len(chunk)
                            if written-last>=128*1024*1024:status(out,'running',stage='download_pinned_assets',file=name,bytes_complete=written,bytes_total=entry['size']);last=written
                if not valid(part,entry):raise RuntimeError('Pinned content digest mismatch')
                part.rename(path);records.append({'file':name,'bytes':entry['size'],'transport':url.split('/')[2]});json_write(out/'download_progress.json',records);break
            except Exception as error:errors.append(str(error));print(json.dumps({'file':name,'transport_failed':template.split('/')[2],'error':str(error)}),flush=True)
        else:raise RuntimeError('All model transports failed: '+json.dumps(errors))
    verified=config|{'all_files_verified':True,'verified_unix':time.time()};json_write(target/'source_verified.json',verified);json_write(out/'download_complete.json',{'state':'complete','model_path':str(target),'records':records,'source':config})
    return target
