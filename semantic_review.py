"""Local sentence embeddings for nearby alternative takes, not factual entailment."""
import re,json
from functools import lru_cache
from app_paths import ROOT,add_dependencies
add_dependencies()
import numpy as np
from speech_safety import facts,lexemes

@lru_cache(maxsize=1)
def runtime():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    folder=ROOT/'.model-cache/semantic-minilm'
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(folder/'onnx/model_quantized.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    tok=Tokenizer.from_file(str(folder/'tokenizer.json'));tok.enable_truncation(max_length=256);tok.enable_padding(pad_id=0,pad_token='[PAD]')
    return session,tok

def encode(texts):
    session,tok=runtime();result=[]
    for start in range(0,len(texts),16):
        batch=tok.encode_batch(texts[start:start+16])
        ids=np.asarray([x.ids for x in batch],dtype=np.int64)
        mask=np.asarray([x.attention_mask for x in batch],dtype=np.int64)
        feed={'input_ids':ids,'attention_mask':mask,'token_type_ids':np.zeros_like(ids)}
        output=session.run(None,{x.name:feed[x.name] for x in session.get_inputs()})[0]
        pooled=(output*mask[:,:,None]).sum(1)/np.maximum(mask.sum(1)[:,None],1)
        result.extend(pooled/np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-9))
    return np.asarray(result)

def terms(text):return lexemes(text)

def opposite(a,b):
    first,second=set(terms(a)),set(terms(b))
    groups=[({'good','great','amazing','better'},{'bad','awful','terrible','worse'}),({'love'},{'hate'}),({'enable'},{'disable'}),({'increase'},{'decrease'})]
    return any((first&x and second&y) or (first&y and second&x) for x,y in groups)

def quality(row):
    words=row['words'];text=row['text'].rstrip()
    complete=not text.endswith(('-', '—','…','...')) and terms(text)[-1] not in {'and','but','to','the','a','because','if'}
    confidence=sum(w.get('probability',1) for w in words)/len(words)
    return (complete,round(confidence,1),min(len(words),40))

def review(rows,ranges,events):
    # Keep the legacy entry point, but never let embedding similarity alone
    # authorize destructive edits. Use the same directional checks as production.
    from contextual_takes import review as contextual_review
    result,report=contextual_review([dict(words=r['words']) for r in rows],ranges,events)
    return result,report['changes'],report
