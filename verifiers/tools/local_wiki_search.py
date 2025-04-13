from pyserini.search.lucene import LuceneSearcher
import json
import gc
import os

_searcher = None
_query_count = 0  

def get_searcher():
    global _searcher, _query_count
    
    if _searcher is None or _query_count >= 50:
        
        if _searcher is not None:
            try:
                _searcher.close()
            except:
                pass
            _searcher = None
            
        
        os.environ["JAVA_OPTS"] = "-Xms4g -Xmx8g -XX:MaxDirectMemorySize=4g -XX:+UseG1GC -XX:+HeapDumpOnOutOfMemoryError -XX:+DisableExplicitGC -XX:UseAVX=0 -XX:+PerfDisableSharedMem"
        gc.collect()  
        _searcher = LuceneSearcher.from_prebuilt_index('wikipedia-kilt-doc')
        _query_count = 0  
        
    return _searcher

def wiki_search(query: str) -> str:
    """Searches Wikipedia and returns relevant article content."""
    try:
        global _query_count
        searcher = get_searcher()
        hits = searcher.search(query, k=1)
        
        if not hits:
            return "No relevant Wikipedia content found."
            
        hit = hits[0]
        doc_id = hit.docid
        doc = searcher.doc(doc_id)
        contents = json.loads(doc.raw())['contents']
        
        _query_count += 1  
        return contents
            
    except Exception as e:
        
        global _searcher
        if _searcher is not None:
            try:
                _searcher.close()
            except:
                pass
            _searcher = None
        return f"Error: searching Wikipedia fails: {str(e)}"
    
if __name__ == "__main__":
    print(wiki_search("What is the capital of France?"))