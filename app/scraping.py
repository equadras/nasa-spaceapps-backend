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
from xml.etree import ElementTree as ET
import html

CSV_URL = "https://raw.githubusercontent.com/jgalazka/SB_publications/main/SB_publication_PMC.csv"


def clean_text(text):
    """
    Clean and normalize text extracted from XML
    Handles special characters, excessive whitespace, and formatting
    """
    if not text:
        return ''
    
    # Decode HTML entities (e.g., &nbsp;, &gt;, etc.)
    text = html.unescape(text)
    
    # Replace multiple newlines with double newline (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace single newlines with spaces (unless it's a paragraph break)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Remove zero-width spaces and other invisible characters
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    
    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)
    
    return text.strip()


def extract_text_from_element(element):
    """
    Extract clean text from XML element, handling inline formatting tags
    Preserves paragraph structure but removes formatting like <italic>, <bold>, etc.
    """
    if element is None:
        return ''
    
    # Get all text content, including from child elements
    text_parts = []
    
    def traverse(elem):
        # Add text before any child elements
        if elem.text:
            text_parts.append(elem.text)
        
        # Process child elements
        for child in elem:
            # For paragraph-like elements, add newlines
            if child.tag in ['p', 'sec', 'title', 'label']:
                if text_parts and not text_parts[-1].endswith('\n'):
                    text_parts.append('\n')
            
            # Recursively process child
            traverse(child)
            
            # Add tail text (text after the closing tag)
            if child.tail:
                text_parts.append(child.tail)
            
            # Add newline after paragraph-like elements
            if child.tag in ['p', 'sec']:
                if text_parts and not text_parts[-1].endswith('\n'):
                    text_parts.append('\n')
    
    traverse(element)
    
    # Join and clean
    text = ''.join(text_parts)
    return clean_text(text)


def extract_section_text(section_elem):
    """
    Extract text from a section, excluding the title
    """
    if section_elem is None:
        return ''
    
    # Remove title from consideration
    text_parts = []
    
    # Get section text but skip the title element
    for elem in section_elem:
        if elem.tag != 'title' and elem.tag != 'label':
            text_parts.append(extract_text_from_element(elem))
    
    return '\n'.join(filter(None, text_parts))


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
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            paper_data = {
                'id': f'PMC{pmc_id}',
                'pmc_link': f'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/'
            }
            
            article = root.find('.//article')
            if article is None:
                return None
            
            # Title - clean it properly
            title_elem = article.find('.//article-title')
            paper_data['title'] = extract_text_from_element(title_elem)
            
            # Authors
            authors = []
            for contrib in article.findall('.//contrib[@contrib-type="author"]'):
                surname = contrib.find('.//surname')
                given = contrib.find('.//given-names')
                if surname is not None:
                    name = clean_text(surname.text) if surname.text else ''
                    if given is not None and given.text:
                        given_clean = clean_text(given.text)
                        name = f"{given_clean} {name}"
                    if name:
                        authors.append(name)
            paper_data['authors'] = ', '.join(authors)
            
            # Year
            pub_date = article.find('.//pub-date[@pub-type="epub"]')
            if pub_date is None:
                pub_date = article.find('.//pub-date[@pub-type="ppub"]')
            if pub_date is not None:
                year_elem = pub_date.find('.//year')
                paper_data['year'] = year_elem.text.strip() if year_elem is not None and year_elem.text else ''
            else:
                paper_data['year'] = ''
            
            # Journal
            journal_elem = article.find('.//journal-title')
            paper_data['journal'] = extract_text_from_element(journal_elem)
            
            # Abstract - improved extraction
            abstract_elem = article.find('.//abstract')
            paper_data['abstract'] = extract_text_from_element(abstract_elem)
            
            # Keywords
            keywords = []
            for kwd in article.findall('.//kwd'):
                kwd_text = extract_text_from_element(kwd)
                if kwd_text:
                    keywords.append(kwd_text)
            paper_data['keywords'] = ', '.join(keywords)
            
            # Body sections
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
                    if title_elem is not None:
                        title_text = extract_text_from_element(title_elem).lower()
                        
                        # Extract section content (excluding title)
                        section_text = extract_section_text(sec)
                        
                        # Classify section by title
                        if 'introduction' in title_text or 'background' in title_text:
                            sections['introduction'] += '\n' + section_text
                        elif 'method' in title_text or 'material' in title_text:
                            sections['methods'] += '\n' + section_text
                        elif 'result' in title_text or 'finding' in title_text:
                            sections['results'] += '\n' + section_text
                        elif 'discussion' in title_text:
                            sections['discussion'] += '\n' + section_text
                        elif 'conclusion' in title_text or 'concluding' in title_text:
                            sections['conclusion'] += '\n' + section_text
                
                # Clean section texts
                for key in sections:
                    sections[key] = clean_text(sections[key])
                
                # Full body text
                paper_data['full_text'] = extract_text_from_element(body)
            else:
                paper_data['full_text'] = ''
            
            paper_data.update(sections)
            
            # Rate limiting
            time.sleep(0.34)
            return paper_data
            
        except Exception as e:
            print(f"\nError fetching PMC{pmc_id}: {str(e)}")
            if attempt < retries - 1:
                print(f"Retrying... (attempt {attempt + 2}/{retries})")
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
    
    print(f"\nProcessing {len(df)} papers...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fetching papers"):
        title_from_csv = row['Title']
        link = row['Link']
        
        pmc_id = extract_pmc_id(link)
        if not pmc_id:
            failed_papers.append({
                'title': title_from_csv,
                'link': link,
                'reason': 'No PMC ID found in link'
            })
            continue
        
        # Fetch using API
        paper_data = fetch_paper_from_ncbi_api(pmc_id)
        
        if paper_data and paper_data.get('title'):
            processed_papers.append(paper_data)
            
            # Save individually
            output_path = processed_dir / f"PMC{pmc_id}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(paper_data, f, indent=2, ensure_ascii=False)
        else:
            failed_papers.append({
                'pmc_id': pmc_id,
                'title': title_from_csv,
                'link': link,
                'reason': 'API fetch failed or no title found'
            })
    
    # Save all papers
    output_file = processed_dir / 'all_papers.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_papers, f, indent=2, ensure_ascii=False)
    
    # Save failed papers
    if failed_papers:
        failed_file = processed_dir / 'failed_papers.json'
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_papers, f, indent=2, ensure_ascii=False)
    
    # Statistics
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Success: {len(processed_papers)}/{len(df)} papers")
    print(f"Failed: {len(failed_papers)}")
    print(f"Saved to: {output_file}")
    
    if processed_papers:
        with_abstract = sum(1 for p in processed_papers if p.get('abstract'))
        with_results = sum(1 for p in processed_papers if p.get('results'))
        with_intro = sum(1 for p in processed_papers if p.get('introduction'))
        with_discussion = sum(1 for p in processed_papers if p.get('discussion'))
        with_conclusion = sum(1 for p in processed_papers if p.get('conclusion'))
        with_authors = sum(1 for p in processed_papers if p.get('authors'))
        
        print(f"\nSTATISTICS:")
        print(f"  Papers with abstract: {with_abstract}/{len(processed_papers)} ({with_abstract/len(processed_papers)*100:.1f}%)")
        print(f"  Papers with introduction: {with_intro}/{len(processed_papers)} ({with_intro/len(processed_papers)*100:.1f}%)")
        print(f"  Papers with results: {with_results}/{len(processed_papers)} ({with_results/len(processed_papers)*100:.1f}%)")
        print(f"  Papers with discussion: {with_discussion}/{len(processed_papers)*100:.1f}%)")
        print(f"  Papers with conclusion: {with_conclusion}/{len(processed_papers)} ({with_conclusion/len(processed_papers)*100:.1f}%)")
        print(f"  Papers with authors: {with_authors}/{len(processed_papers)} ({with_authors/len(processed_papers)*100:.1f}%)")
    
    if failed_papers:
        print(f"\nFailed papers saved to: {processed_dir / 'failed_papers.json'}")
    
    print(f"{'='*70}\n")
    
    return processed_papers, failed_papers


def main():
    print("NASA BIOSCIENCE PAPERS - DOWNLOAD AND PROCESSING")
    print("="*70)
    
    df = download_csv()
    print(f"Downloaded CSV with {len(df)} papers")
    
    processed_papers, failed_papers = process_all_papers(df)
    
    print("\nReady to load into ChromaDB!")

if __name__ == "__main__":
    main()
