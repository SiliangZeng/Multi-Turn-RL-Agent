import os


os.environ["JAVA_OPTS"] = "-Xms16g -Xmx32g " "-XX:MaxDirectMemorySize=16g "

from pyserini.search.lucene import LuceneSearcher
import json


_searcher = LuceneSearcher.from_prebuilt_index("wikipedia-kilt-doc")


def wiki_search(query: str) -> str:
    """Searches Wikipedia and returns the top matching article content."""

    try:
        hits = _searcher.search(query, k=1)
        if not hits:
            return "No relevant Wikipedia content found."

        doc = _searcher.doc(hits[0].docid)
        contents = json.loads(doc.raw())["contents"]

        return contents

    except Exception as e:
        return f"Error: during search: {e}"


if __name__ == "__main__":
    print(wiki_search("What is the capital of France?"))
