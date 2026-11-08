"""CPU tokenizer replay, independent DF counting and supervised-file denial."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def main(args):
    out=Path(args.out);cfg=json.loads((out/'config.json').read_text());folder=out/'contexts'/args.context;texts=json.loads((out/'encoder_inputs.json').read_text());original_load=np.load;denied=[]
    def guarded(path,*a,**kw):
        if Path(path).name in ['dataset.npz','auxiliary_targets.npz']:denied.append(str(path));raise RuntimeError('Supervised corpus access prohibited during independent tokenization')
        return original_load(path,*a,**kw)
    np.load=guarded;fit=np.load(folder/'idf_fit_positions.npz')['positions'];index=np.load(folder/'token_index.npz');pairs=np.load(folder/'pair_positions.npz');expected=np.unique(np.r_[pairs['queries'],pairs['documents']]);np.testing.assert_array_equal(expected,index['positions']);tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True,trust_remote_code=False,padding_side='left');allpositions=np.unique(np.r_[fit,expected]);encoded=tok([texts[int(pos)] for pos in allpositions],add_special_tokens=True,truncation=True,max_length=cfg['max_length'],padding=False)['input_ids'];bypos=dict(zip(map(int,allpositions),encoded));counts=Counter()
    for pos in fit:counts.update(set(bypos[int(pos)]))
    saved=json.loads((folder/'idf.json').read_text());assert saved['documents']==len(fit);assert {int(k):v for k,v in saved['counts'].items()}==dict(counts);special=set(tok.all_special_ids);eligibility={}
    for i,pos in enumerate(expected):
        tokens=[]
        for token in bypos[int(pos)]:
            if token not in eligibility:eligibility[token]=token not in special and any(ch.isalnum() for ch in tok.decode([token],skip_special_tokens=False))
            if eligibility[token]:tokens.append(token)
        a,b=index['offsets'][i:i+2];np.testing.assert_array_equal(index['ids'][a:b],tokens);weights=np.array([np.log((len(fit)+1)/(counts.get(token,0)+1))+1 for token in tokens]);np.testing.assert_array_equal(index['weights'][a:b],weights)
    result={'state':'passed','ownfit_DF_rows':len(fit),'encoded_evaluation_documents':len(expected),'all_cached_token_ids_and_IDF_weights_verified':True,'independent_Counter_and_batch_tokenizer':True,'supervised_file_access_forbidden':True,'denied_access_attempts':denied};(folder/'tokenization_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--context',required=True);main(p.parse_args())
