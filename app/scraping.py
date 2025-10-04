"""
Download e processamento CORRETO dos papers NASA
Usa NCBI E-utilities API para dados estruturados
"""
import pandas as pd
import requests
from pathlib import Path
import json
import time
from tqdm import tqdm
import re
import xmltodict
from xml.etree import ElementTree as ET

CSV_URL = "https://raw.githubusercontent.com/jgalazka/SB_publications/main/SB_publication_PMC.csv"

def download_csv():
    """Baixa o CSV do GitHub"""
    
    Path('data').mkdir(exist_ok=True)
    response = requests.get(CSV_URL)
    csv_path = Path('data/papers_list.csv')
    
    with open(csv_path, 'wb') as f:
        f.write(response.content)
    
    df = pd.read_csv(csv_path)
    return df

def extract_pmc_id(link):
    """Extrai PMC ID do link"""
    match = re.search(r'PMC(\d+)', link)
    return match.group(1) if match else None

def fetch_paper_from_ncbi_api(pmc_id, retries=3):
    """
    Busca paper usando NCBI E-utilities API
    Retorna dados estruturados em XML
    """
    for attempt in range(retries):
        try:
            # URL da API do NCBI
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            params = {
                'db': 'pmc',
                'id': pmc_id,
                'retmode': 'xml'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            xml_content = response.read()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            paper_data = {
                'id': f'PMC{pmc_id}',
                'pmc_link': f'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/'
            }
            
            article = root.find('.//article')
            if article is None:
                return None
            
            title_elem = article.find('.//article-title')
            paper_data['title'] = title_elem.text if title_elem is not None else ''
            
            # Autores
            authors = []
            for contrib in article.findall('.//contrib[@contrib-type="author"]'):
                surname = contrib.find('.//surname')
                given = contrib.find('.//given-names')
                if surname is not None:
                    name = surname.text
                    if given is not None:
                        name = f"{given.text} {name}"
                    authors.append(name)
            paper_data['authors'] = ', '.join(authors)
            
            # Ano
            pub_date = article.find('.//pub-date[@pub-type="epub"]')
            if pub_date is None:
                pub_date = article.find('.//pub-date[@pub-type="ppub"]')
            if pub_date is not None:
                year_elem = pub_date.find('.//year')
                paper_data['year'] = year_elem.text if year_elem is not None else ''
            else:
                paper_data['year'] = ''
            
            # Journal
            journal_elem = article.find('.//journal-title')
            paper_data['journal'] = journal_elem.text if journal_elem is not None else ''
            
            # Abstract
            abstract_elem = article.find('.//abstract')
            if abstract_elem is not None:
                abstract_text = ' '.join(abstract_elem.itertext())
                paper_data['abstract'] = abstract_text.strip()[:5000]
            else:
                paper_data['abstract'] = ''
            
            # Keywords
            keywords = []
            for kwd in article.findall('.//kwd'):
                if kwd.text:
                    keywords.append(kwd.text)
            paper_data['keywords'] = ', '.join(keywords)
            
            body = article.find('.//body')
            
            sections = {
                'introduction': '',
                'methods': '',
                'results': '',
                'discussion': '',
                'conclusion': ''
            }
            
            if body is not None:
                for sec in body.findall('.//sec'):
                    title_elem = sec.find('.//title')
                    if title_elem is not None and title_elem.text:
                        title_text = title_elem.text.lower()
                        
                        # Extrair texto da seção
                        section_text = ' '.join(sec.itertext()).strip()
                        
                        if 'introduction' in title_text:
                            sections['introduction'] = section_text[:3000]
                        elif 'method' in title_text or 'material' in title_text:
                            sections['methods'] = section_text[:3000]
                        elif 'result' in title_text:
                            sections['results'] = section_text[:4000]
                        elif 'discussion' in title_text:
                            sections['discussion'] = section_text[:3000]
                        elif 'conclusion' in title_text:
                            sections['conclusion'] = section_text[:2000]
                
                # Texto completo do body
                full_body = ' '.join(body.itertext()).strip()
                paper_data['full_text'] = full_body[:20000]
            else:
                paper_data['full_text'] = ''
            
            paper_data.update(sections)
            
            time.sleep(0.34)
            return paper_data
            
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            else:
                return None
    
    return None

def process_all_papers(df):
    """Processa todos os papers"""
    
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    processed_papers = []
    failed_papers = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        title_from_csv = row['Title']
        link = row['Link']
        
        pmc_id = extract_pmc_id(link)
        if not pmc_id:
            failed_papers.append({'title': title_from_csv, 'link': link, 'reason': 'No PMC ID'})
            continue
        
        # Fetch usando API
        paper_data = fetch_paper_from_ncbi_api(pmc_id)
        
        if paper_data and paper_data.get('title'):
            processed_papers.append(paper_data)
            
            # Salvar individualmente
            output_path = processed_dir / f"PMC{pmc_id}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(paper_data, f, indent=2, ensure_ascii=False)
        else:
            failed_papers.append({
                'pmc_id': pmc_id,
                'title': title_from_csv,
                'link': link,
                'reason': 'API fetch failed'
            })
    
    # Salvar tudo
    output_file = processed_dir / 'all_papers.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_papers, f, indent=2, ensure_ascii=False)
    
    if failed_papers:
        failed_file = processed_dir / 'failed_papers.json'
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_papers, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessamento concluído!")
    print(f"Sucesso: {len(processed_papers)}/{len(df)} papers")
    print(f"Falhas: {len(failed_papers)}")
    print(f"Salvos em: {output_file}")
    
    if processed_papers:
        with_abstract = sum(1 for p in processed_papers if p.get('abstract'))
        with_results = sum(1 for p in processed_papers if p.get('results'))
        with_authors = sum(1 for p in processed_papers if p.get('authors'))
        
        print(f"\nESTATÍSTICAS:")
        print(f"   Papers com abstract: {with_abstract}/{len(processed_papers)}")
        print(f"   Papers com results: {with_results}/{len(processed_papers)}")
        print(f"   Papers com authors: {with_authors}/{len(processed_papers)}")
    
    return processed_papers, failed_papers

def main():
    
    df = download_csv()
    processed_papers, failed_papers = process_all_papers(df)
    
    print("\nProcesso concluído! Pronto para carregar no ChromaDB.")

if __name__ == "__main__":
    main()

