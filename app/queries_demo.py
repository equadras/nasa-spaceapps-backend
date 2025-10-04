# scripts/3_query_interactive.py
"""
Query interativo - Busca papers por similaridade via input do usuário
"""
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
import sys

class PaperQueryEngine:
    def __init__(self):
        """Inicializa o engine de busca"""
        print("🔧 Carregando ChromaDB e modelo de embeddings...")
        
        # ChromaDB
        db_path = Path('../database/chroma_db')
        if not db_path.exists():
            print("❌ Database não encontrado!")
            print("Execute primeiro: python3 script/2_load_to_chromadb.py")
            sys.exit(1)
        
        self.client = chromadb.PersistentClient(path=str(db_path))
        
        try:
            self.collection = self.client.get_collection("nasa_bioscience")
        except Exception as e:
            print(f"❌ Collection 'nasa_bioscience' não encontrada: {e}")
            sys.exit(1)
        
        # Modelo de embeddings
        print("📥 Carregando modelo de embeddings...")
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Stats
        self.total_chunks = self.collection.count()
        print(f"✅ Pronto! {self.total_chunks} chunks disponíveis\n")
    
    def query(self, question, top_k=10):
        """Busca papers por similaridade"""
        
        # Gerar embedding da pergunta
        query_embedding = self.model.encode(question).tolist()
        
        # Buscar no ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results
    
    def display_results(self, question, results):
        """Exibe resultados de forma formatada"""
        
        print(f"\n{'=' * 80}")
        print(f"🔍 Query: {question}")
        print('=' * 80)
        
        if not results['ids'][0]:
            print("\n❌ Nenhum resultado encontrado.")
            return
        
        # Agrupar por paper_id (pode ter múltiplos chunks do mesmo paper)
        papers_dict = {}
        
        for i, (chunk_id, metadata, distance) in enumerate(zip(
            results['ids'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            paper_id = metadata.get('paper_id', chunk_id)
            
            if paper_id not in papers_dict:
                papers_dict[paper_id] = {
                    'metadata': metadata,
                    'best_score': 1 - distance,  # Converter distância para score
                    'chunk_count': 1
                }
            else:
                papers_dict[paper_id]['chunk_count'] += 1
                # Manter o melhor score
                score = 1 - distance
                if score > papers_dict[paper_id]['best_score']:
                    papers_dict[paper_id]['best_score'] = score
        
        # Ordenar por score
        sorted_papers = sorted(
            papers_dict.items(),
            key=lambda x: x[1]['best_score'],
            reverse=True
        )
        
        print(f"\n📚 Top {len(sorted_papers)} papers mais relevantes:\n")
        
        for i, (paper_id, data) in enumerate(sorted_papers[:10], 1):
            meta = data['metadata']
            score = data['best_score']
            chunks = data['chunk_count']
            
            print(f"{i}. {meta.get('title', 'N/A')}")
            print(f"   📊 Relevância: {score:.4f} ({chunks} chunk{'s' if chunks > 1 else ''} encontrado{'s' if chunks > 1 else ''})")
            
            if meta.get('authors'):
                authors = meta['authors'][:80]
                print(f"   👥 Autores: {authors}{'...' if len(meta['authors']) > 80 else ''}")
            
            if meta.get('year'):
                print(f"   📅 Ano: {meta['year']}")
            
            if meta.get('journal'):
                journal = meta['journal'][:60]
                print(f"   📰 Journal: {journal}{'...' if len(meta['journal']) > 60 else ''}")
            
            if meta.get('keywords'):
                keywords = meta['keywords'][:80]
                print(f"   🏷️  Keywords: {keywords}{'...' if len(meta['keywords']) > 80 else ''}")
            
            if meta.get('pmc_link'):
                print(f"   🔗 Link: {meta['pmc_link']}")
            
            print()
        
        print('─' * 80)
    
    def show_help(self):
        """Mostra ajuda"""
        print("\n" + "=" * 80)
        print("💡 COMANDOS DISPONÍVEIS")
        print("=" * 80)
        print("\n📝 Digite sua pergunta em inglês sobre biociências espaciais")
        print("\nExemplos de perguntas:")
        print("  • What are the effects of microgravity on bone density?")
        print("  • How does space radiation affect DNA repair mechanisms?")
        print("  • What are the challenges for plant growth in microgravity?")
        print("  • How does spaceflight affect the immune system?")
        print("  • What biological adaptations are needed for Mars missions?")
        print("\n⚙️  Comandos especiais:")
        print("  • help  - Mostra esta ajuda")
        print("  • stats - Mostra estatísticas do banco")
        print("  • quit  - Sair")
        print("=" * 80)
    
    def show_stats(self):
        """Mostra estatísticas"""
        print("\n" + "=" * 80)
        print("📊 ESTATÍSTICAS DO BANCO DE DADOS")
        print("=" * 80)
        
        # Pegar amostra
        sample = self.collection.peek(limit=100)
        
        # Contar papers únicos
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
        
        print(f"\n💾 Total de chunks: {self.total_chunks}")
        print(f"📚 Papers únicos (amostra): ~{len(unique_papers)} (de 100 chunks)")
        print(f"📚 Estimativa total de papers: ~607")
        
        if years:
            print(f"📅 Intervalo de anos (amostra): {min(years)} - {max(years)}")
        
        print("\n✅ Banco carregado e pronto para queries!")
        print("=" * 80)

def main():
    print("=" * 80)
    print("🚀 NASA BIOSCIENCE PAPERS - INTERACTIVE QUERY")
    print("=" * 80)
    print()
    
    # Inicializar engine
    engine = PaperQueryEngine()
    
    # Mostrar ajuda inicial
    print("💡 Digite 'help' para ver comandos ou comece a fazer perguntas!")
    print("💡 Digite 'quit' para sair")
    print()
    
    # Loop interativo
    while True:
        try:
            # Ler input
            question = input("❓ Sua pergunta: ").strip()
            
            # Comandos vazios
            if not question:
                continue
            
            # Comando: quit
            if question.lower() in ['quit', 'exit', 'q', 'sair']:
                print("\n👋 Até logo!")
                break
            
            # Comando: help
            if question.lower() in ['help', 'ajuda', '?']:
                engine.show_help()
                continue
            
            # Comando: stats
            if question.lower() in ['stats', 'status', 'info']:
                engine.show_stats()
                continue
            
            # Query normal
            print("\n🔍 Buscando...")
            results = engine.query(question, top_k=20)
            engine.display_results(question, results)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrompido pelo usuário. Até logo!")
            break
        
        except Exception as e:
            print(f"\n❌ Erro: {e}")
            print("💡 Tente novamente ou digite 'help' para ajuda")

if __name__ == "__main__":
    main()
