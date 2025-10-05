"""
Hybrid Query Engine - Vector + BM25 with similarity threshold
Only returns results above minimum similarity score
"""
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
import sys
import json
from datetime import datetime
import numpy as np
from rank_bm25 import BM25Okapi

class HybridPaperQueryEngine:
    def __init__(self, alpha=0.80, similarity_threshold=0.3):
        """
        Initialize hybrid search engine with threshold filtering
        
        Args:
            alpha: Weight for vector search (0-1)
            similarity_threshold: Minimum similarity to return results (0-1)
        """
        print("Loading ChromaDB and embedding model...")
        
        self.alpha = alpha
        self.similarity_threshold = similarity_threshold
        
        # ChromaDB
        db_path = Path('../database/chroma_db')
        if not db_path.exists():
            print("ERROR: Database not found!")
            sys.exit(1)
        
        self.client = chromadb.PersistentClient(path=str(db_path))
        
        try:
            self.collection = self.client.get_collection("nasa_bioscience")
        except Exception as e:
            print(f"ERROR: Collection not found: {e}")
            sys.exit(1)
        
        # Embedding model
        print("Loading embedding model...")
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        # self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Build BM25 index
        print("Building BM25 index...")
        self._build_bm25_index()
        
        self.total_chunks = self.collection.count()
        print(f"Ready! {self.total_chunks} chunks available")
        print(f"Hybrid: {self.alpha*100:.0f}% vector + {(1-self.alpha)*100:.0f}% BM25")
        print(f"Similarity threshold: {self.similarity_threshold:.2f}\n")
    
    def _build_bm25_index(self):
        """Build BM25 index from all documents"""
        all_data = self.collection.get(include=['documents', 'metadatas'])
        
        self.all_ids = all_data['ids']
        self.all_documents = all_data['documents']
        self.all_metadatas = all_data['metadatas']
        
        tokenized_corpus = [doc.lower().split() for doc in self.all_documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.doc_map = {doc_id: idx for idx, doc_id in enumerate(self.all_ids)}
        
        print(f"BM25 index built with {len(self.all_ids)} chunks")
    
    def query(self, question, top_k=20):
        """
        Hybrid search with threshold filtering
        Returns empty if no results meet threshold
        """
        # 1. Vector search
        query_embedding = self.model.encode(question).tolist()
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        vector_scores = {}
        for doc_id, distance in zip(vector_results['ids'][0], vector_results['distances'][0]):
            vector_scores[doc_id] = 1 - distance
        
        # 2. BM25 search
        tokenized_query = question.lower().split()
        bm25_scores_raw = self.bm25.get_scores(tokenized_query)
        max_bm25 = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1
        bm25_scores = bm25_scores_raw / max_bm25
        
        # 3. Combine scores
        combined_scores = {}
        for doc_id in self.all_ids:
            idx = self.doc_map[doc_id]
            score = 0.0
            
            if doc_id in vector_scores:
                score += self.alpha * vector_scores[doc_id]
            score += (1 - self.alpha) * bm25_scores[idx]
            
            # Only include if above threshold
            if score >= self.similarity_threshold:
                combined_scores[doc_id] = score
        
        # 4. If nothing meets threshold, return empty
        if not combined_scores:
            return {
                'ids': [[]],
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
        
        # 5. Sort and build results
        sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        final_ids = [doc_id for doc_id, _ in sorted_ids]
        final_scores = [score for _, score in sorted_ids]
        
        return {
            'ids': [final_ids],
            'documents': [[self.all_documents[self.doc_map[doc_id]] for doc_id in final_ids]],
            'metadatas': [[self.all_metadatas[self.doc_map[doc_id]] for doc_id in final_ids]],
            'distances': [[1 - score for score in final_scores]]
        }
    
    def display_results(self, question, results):
        """Display results grouped by paper"""
        print(f"\n{'=' * 80}")
        print(f"Query: {question}")
        print(f"Threshold: {self.similarity_threshold:.2f} | Hybrid: {self.alpha*100:.0f}% vector + {(1-self.alpha)*100:.0f}% BM25")
        print('=' * 80)
        
        if not results['ids'][0]:
            print("\nNo results found above similarity threshold.")
            print(f"Try lowering the threshold (current: {self.similarity_threshold:.2f})")
            return
        
        # Group by paper_id
        papers_dict = {}
        for chunk_id, metadata, distance in zip(
            results['ids'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            paper_id = metadata.get('paper_id', chunk_id)
            score = 1 - distance
            
            if paper_id not in papers_dict:
                papers_dict[paper_id] = {
                    'metadata': metadata,
                    'best_score': score,
                    'chunk_count': 1
                }
            else:
                papers_dict[paper_id]['chunk_count'] += 1
                if score > papers_dict[paper_id]['best_score']:
                    papers_dict[paper_id]['best_score'] = score
        
        # Sort by score
        sorted_papers = sorted(
            papers_dict.items(),
            key=lambda x: x[1]['best_score'],
            reverse=True
        )
        
        print(f"\nTop {len(sorted_papers)} papers (above threshold):\n")
        
        for i, (paper_id, data) in enumerate(sorted_papers[:10], 1):
            meta = data['metadata']
            score = data['best_score']
            chunks = data['chunk_count']
            
            print(f"{i}. {meta.get('title', 'N/A')}")
            print(f"   Relevance: {score:.4f} ({chunks} chunk{'s' if chunks > 1 else ''})")
            
            if meta.get('authors'):
                print(f"   Authors: {meta['authors'][:80]}")
            if meta.get('year'):
                print(f"   Year: {meta['year']}")
            if meta.get('journal'):
                print(f"   Journal: {meta['journal'][:60]}")
            if meta.get('keywords'):
                print(f"   Keywords: {meta['keywords'][:80]}")
            if meta.get('pmc_link'):
                print(f"   Link: {meta['pmc_link']}")
            print()
        
        print('-' * 80)
    
    def change_threshold(self):
        """Change similarity threshold interactively"""
        print(f"\nCurrent threshold: {self.similarity_threshold:.2f}")
        print("Results must score above this to be returned")
        
        try:
            new_threshold = float(input("\nEnter new threshold (0.0-1.0): "))
            if 0 <= new_threshold <= 1:
                self.similarity_threshold = new_threshold
                print(f"Updated threshold to: {self.similarity_threshold:.2f}")
            else:
                print("Invalid. Must be 0.0-1.0")
        except ValueError:
            print("Invalid input")
    
    def change_weights(self):
        """Change vector/BM25 weights"""
        print(f"\nCurrent: {self.alpha*100:.0f}% vector + {(1-self.alpha)*100:.0f}% BM25")
        
        try:
            new_alpha = float(input("\nEnter vector weight (0-100): ")) / 100
            if 0 <= new_alpha <= 1:
                self.alpha = new_alpha
                print(f"Updated: {self.alpha*100:.0f}% vector + {(1-self.alpha)*100:.0f}% BM25")
            else:
                print("Invalid. Must be 0-100")
        except ValueError:
            print("Invalid input")
    
    def show_help(self):
        """Show help"""
        print("\n" + "=" * 80)
        print("HYBRID SEARCH WITH THRESHOLD FILTERING")
        print("=" * 80)
        print("\nJust type your question to search")
        print("\nSpecial commands:")
        print("  help      - Show this help")
        print("  threshold - Change minimum similarity threshold")
        print("  weight    - Change vector/BM25 weights")
        print("  stats     - Show statistics")
        print("  quit      - Exit")
        print("\nNote: Only results above the similarity threshold are shown")
        print(f"Current threshold: {self.similarity_threshold:.2f}")
        print("=" * 80)
    
    def show_stats(self):
        """Show statistics"""
        print("\n" + "=" * 80)
        print("DATABASE STATISTICS")
        print("=" * 80)
        
        sample = self.collection.peek(limit=100)
        unique_papers = set()
        years = []
        
        for meta in sample['metadatas']:
            if meta.get('paper_id'):
                unique_papers.add(meta['paper_id'])
            if meta.get('year'):
                try:
                    years.append(int(meta['year']))
                except:
                    pass
        
        print(f"\nTotal chunks: {self.total_chunks}")
        print(f"Unique papers (sample): ~{len(unique_papers)}")
        print(f"Estimated total papers: ~607")
        if years:
            print(f"Year range: {min(years)}-{max(years)}")
        
        print(f"\nCurrent settings:")
        print(f"  Threshold: {self.similarity_threshold:.2f}")
        print(f"  Weights: {self.alpha*100:.0f}% vector + {(1-self.alpha)*100:.0f}% BM25")
        print("=" * 80)


def main():
    print("=" * 80)
    print("NASA BIOSCIENCE - HYBRID SEARCH WITH THRESHOLD")
    print("=" * 80)
    print()
    
    # Initialize with threshold=0.3 (only show results with >30% similarity)
    engine = HybridPaperQueryEngine(alpha=0.4, similarity_threshold=0.3)
    
    print("Type 'help' for commands")
    print()
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if question.lower() in ['help', '?']:
                engine.show_help()
                continue
            
            if question.lower() == 'threshold':
                engine.change_threshold()
                continue
            
            if question.lower() == 'weight':
                engine.change_weights()
                continue
            
            if question.lower() in ['stats', 'status']:
                engine.show_stats()
                continue
            
            # Query
            print("\nSearching...")
            results = engine.query(question, top_k=20)
            engine.display_results(question, results)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
