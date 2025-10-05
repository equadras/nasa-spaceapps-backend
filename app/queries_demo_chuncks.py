"""
Query interativo - Busca papers por similaridade via input do usuário
"""
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
import sys
import json
from datetime import datetime

class PaperQueryEngine:
    def __init__(self):
        """Inicializa o engine de busca"""
        print("Loading ChromaDB and embedding model...")
        
        # ChromaDB
        db_path = Path('../database/chroma_db')
        if not db_path.exists():
            print("ERROR: Database not found!")
            print("Run first: python3 script/2_load_to_chromadb.py")
            sys.exit(1)
        
        self.client = chromadb.PersistentClient(path=str(db_path))
        
        try:
            self.collection = self.client.get_collection("nasa_bioscience")
        except Exception as e:
            print(f"ERROR: Collection 'nasa_bioscience' not found: {e}")
            sys.exit(1)
        
        # Modelo de embeddings (use the same as in indexing!)
        print("Loading embedding model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Stats
        self.total_chunks = self.collection.count()
        print(f"Ready! {self.total_chunks} chunks available\n")
    
    def query(self, question, top_k=10):
        """Busca papers por similaridade"""
        
        # Gerar embedding da pergunta
        query_embedding = self.model.encode(question).tolist()
        
        # Buscar no ChromaDB - IMPORTANT: include 'documents' to get chunk text
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']  # Added documents!
        )
        
        return results
    
    def save_chunks_to_file(self, question, results, filename='chunkTxt.txt'):
        """Save chunk texts to a file"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(filename, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*80 + "\n")
            f.write(f"NASA BIOSCIENCE - QUERY RESULTS\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Query: {question}\n")
            f.write("="*80 + "\n\n")
            
            # Check if results exist
            if not results['ids'][0]:
                f.write("No results found.\n")
                return
            
            # Write each chunk
            for i, (chunk_id, doc_text, metadata, distance) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],  # This contains the chunk text!
                results['metadatas'][0],
                results['distances'][0]
            ), 1):
                score = 1 - distance
                
                f.write(f"\n{'='*80}\n")
                f.write(f"CHUNK {i}\n")
                f.write(f"{'='*80}\n\n")
                
                # Metadata
                f.write(f"Score: {score:.4f}\n")
                f.write(f"Paper ID: {metadata.get('paper_id', 'N/A')}\n")
                f.write(f"Title: {metadata.get('title', 'N/A')}\n")
                f.write(f"Authors: {metadata.get('authors', 'N/A')}\n")
                f.write(f"Year: {metadata.get('year', 'N/A')}\n")
                f.write(f"Journal: {metadata.get('journal', 'N/A')}\n")
                
                if metadata.get('keywords'):
                    f.write(f"Keywords: {metadata['keywords']}\n")
                
                if metadata.get('pmc_link'):
                    f.write(f"Link: {metadata['pmc_link']}\n")
                
                # Chunk text
                f.write(f"\n{'-'*80}\n")
                f.write(f"CHUNK TEXT:\n")
                f.write(f"{'-'*80}\n\n")
                f.write(doc_text)
                f.write("\n\n")
        
        print(f"\nChunks saved to: {filename}")
    
    def save_chunks_to_json(self, question, results, filename='chunkTxt.json'):
        """Save chunks to JSON format (alternative)"""
        
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'query': question,
            'total_results': len(results['ids'][0]) if results['ids'] else 0,
            'chunks': []
        }
        
        if results['ids'][0]:
            for chunk_id, doc_text, metadata, distance in zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                chunk_data = {
                    'chunk_id': chunk_id,
                    'score': float(1 - distance),
                    'text': doc_text,
                    'metadata': metadata
                }
                output_data['chunks'].append(chunk_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Chunks saved to: {filename}")
    
    def display_results(self, question, results):
        """Exibe resultados de forma formatada"""
        
        print(f"\n{'=' * 80}")
        print(f"Query: {question}")
        print('=' * 80)
        
        if not results['ids'][0]:
            print("\nNo results found.")
            return
        
        # Show chunk previews
        print(f"\nTop {len(results['ids'][0])} chunks:\n")
        
        for i, (chunk_id, doc_text, metadata, distance) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            score = 1 - distance
            
            print(f"[{i}] {metadata.get('title', 'N/A')[:70]}...")
            print(f"    Score: {score:.4f} | Year: {metadata.get('year', 'N/A')}")
            
            # Show chunk preview (first 200 chars)
            preview = doc_text[:200].replace('\n', ' ')
            print(f"    Preview: {preview}...")
            print()
        
        print('-' * 80)
    
    def show_help(self):
        """Mostra ajuda"""
        print("\n" + "=" * 80)
        print("AVAILABLE COMMANDS")
        print("=" * 80)
        print("\nType your question in English about space biosciences")
        print("\nExample questions:")
        print("  • What are the effects of microgravity on bone density?")
        print("  • How does space radiation affect DNA repair mechanisms?")
        print("  • What are the challenges for plant growth in microgravity?")
        print("  • How does spaceflight affect the immune system?")
        print("\nSpecial commands:")
        print("  • help  - Show this help")
        print("  • stats - Show database statistics")
        print("  • quit  - Exit")
        print("=" * 80)
    
    def show_stats(self):
        """Mostra estatísticas"""
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
        print(f"Unique papers (sample): ~{len(unique_papers)} (from 100 chunks)")
        print(f"Estimated total papers: ~607")
        
        if years:
            print(f"Year range (sample): {min(years)} - {max(years)}")
        
        print("\nDatabase loaded and ready!")
        print("=" * 80)

def main():
    print("=" * 80)
    print("NASA BIOSCIENCE PAPERS - INTERACTIVE QUERY")
    print("=" * 80)
    print()
    
    engine = PaperQueryEngine()
    
    print("Type 'help' for commands or start asking questions!")
    print("Type 'quit' to exit")
    print()
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q', 'sair']:
                print("\nGoodbye!")
                break
            
            if question.lower() in ['help', 'ajuda', '?']:
                engine.show_help()
                continue
            
            if question.lower() in ['stats', 'status', 'info']:
                engine.show_stats()
                continue
            
            # Query
            print("\nSearching...")
            results = engine.query(question, top_k=10)
            
            # Display results
            engine.display_results(question, results)
            
            # Save chunks to file
            engine.save_chunks_to_file(question, results, filename='chunkTxt.txt')
            
            # Optional: also save as JSON
            engine.save_chunks_to_json(question, results, filename='chunkJson.json')
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            print("Try again or type 'help'")

if __name__ == "__main__":
    main()
