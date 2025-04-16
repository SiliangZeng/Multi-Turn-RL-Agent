from pyserini.search.lucene import LuceneSearcher
import json
import gc
import os
import subprocess
import time

_searcher = None
_query_count = 0  

def setup_system_limits():
    """尝试修改系统级内存限制"""
    try:
        # 增加内存映射限制，Lucene需要大量连续内存
        subprocess.run("sysctl -w vm.max_map_count=1048576", shell=True)
        
        # 移除虚拟内存限制
        os.system("ulimit -v unlimited")
        os.system("ulimit -m unlimited")
    except Exception as e:
        print(f"Warning: Failed to set system limits: {e}")

def get_searcher():
    global _searcher, _query_count
    
    if _searcher is None or _query_count >= 50:
        
        if _searcher is not None:
            try:
                _searcher.close()
            except:
                pass
            _searcher = None
        
        # 先尝试设置系统限制
        setup_system_limits()
        
        # 加载前主动进行垃圾回收
        gc.collect()
        
        os.environ["JAVA_OPTS"] = "-Xms16g -Xmx32g -XX:MaxDirectMemorySize=16g -XX:+UseG1GC -XX:+HeapDumpOnOutOfMemoryError -XX:+DisableExplicitGC -XX:G1ReservePercent=15"
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
        
        # Only run garbage collection occasionally to reduce overhead
        if _query_count % 20 == 0:
            gc.collect()
            
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