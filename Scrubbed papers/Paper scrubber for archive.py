import os
import re
import csv
import time
import requests
import PyPDF2
from difflib import SequenceMatcher

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file"""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_paper_info(text):
    """
    Extract paper information from text with numbered reference format
    Specifically handles formats like:
    "1. Author1, Author2, et al. (YEAR). Title. Journal details."
    """
    # Pattern to match numbered references
    # Look for patterns like "1." or "1 " at the beginning of lines or after newlines
    paper_entries = re.findall(r'(?:^|\n)\s*\d+\s*\.?\s+(.+?)(?=(?:\n\s*\d+\s*\.?\s+)|$)', text, re.DOTALL)
    
    papers = []
    for entry in paper_entries:
        entry = entry.strip()
        if len(entry) < 30:  # Skip very short entries
            continue
        
        # Try to extract author, year, title and journal
        # Pattern: Authors (YEAR). Title. Journal.
        match = re.search(r'(.+?)\s*\((\d{4})\)\.\s*(.+?)\.\s*([^\.]+\.)(?:\s*|$)', entry)
        
        if match:
            authors = match.group(1).strip()
            year = match.group(2).strip()
            title = match.group(3).strip()
            journal_info = match.group(4).strip()
            
            papers.append({
                'title': title,
                'authors': authors,
                'year': year,
                'journal_info': journal_info
            })
        else:
            # Alternative pattern without clear separation
            # Try to find author part (often contains commas and ends with year in parentheses)
            author_year_match = re.search(r'(.+?)\s*\((\d{4})\)', entry)
            if author_year_match:
                authors = author_year_match.group(1).strip()
                year = author_year_match.group(2).strip()
                
                # Rest of the text might contain title and journal
                remaining = entry[author_year_match.end():].strip()
                
                # Title often starts with a period after the year and ends with another period
                title_match = re.search(r'^\.\s*(.+?)\.\s*', remaining)
                if title_match:
                    title = title_match.group(1).strip()
                    journal_info = remaining[title_match.end():].strip()
                else:
                    # If we can't clearly identify the title, use a best guess approach
                    # Split remaining text at the last period before any journal-like words
                    journal_match = re.search(r'(.+)(\.\s*(?:Journal|PLoS|Proceedings|Malaria|Biology|Science|Nature).*$)', remaining)
                    if journal_match:
                        title = journal_match.group(1).strip() + '.'
                        journal_info = journal_match.group(2).strip()
                    else:
                        # Last resort: just split at the last period
                        parts = remaining.split('.')
                        if len(parts) > 1:
                            title = '.'.join(parts[:-1]).strip()
                            journal_info = parts[-1].strip()
                        else:
                            title = remaining
                            journal_info = ""
                
                papers.append({
                    'title': title,
                    'authors': authors,
                    'year': year,
                    'journal_info': journal_info
                })
    
    return papers

def search_crossref_by_title(title):
    """Search CrossRef API for paper info using title
    Returns DOI if found"""
    url = "https://api.crossref.org/works"
    
    # Clean title for better matching
    query = title.replace(':', ' ').replace('-', ' ').strip()
    
    params = {
        'query.title': query,
        'rows': 5,
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['status'] == 'ok' and data['message']['items']:
            items = data['message']['items']
            
            # Match checking with title similarity
            best_match = None
            best_score = 0
            
            for item in items:
                if 'title' in item and item['title']:
                    item_title = item['title'][0]
                    score = SequenceMatcher(None, title.lower(), item_title.lower()).ratio()
                    
                    if score > best_score and score > 0.65:  # Reduced threshold for better matching
                        best_score = score
                        best_match = item
            
            if best_match and 'DOI' in best_match:
                return {
                    'doi': best_match['DOI'],
                    'match_score': best_score,
                    'matched_title': best_match['title'][0] if 'title' in best_match and best_match['title'] else None,
                    'publisher': best_match.get('publisher')
                }
    
    except Exception as e:
        print(f"Error searching CrossRef: {e}")
    
    return None

def search_pubmed_by_title_and_author(title, author=None):
    """Search PubMed API for paper info using title and first author
    Returns PMID if found"""
    
    # First search for the article
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}esearch.fcgi"
    
    # Create search term with title and optionally first author's last name
    search_term = title
    if author:
        # Extract first author's last name if multiple authors
        first_author = author.split(',')[0].strip()
        last_name = first_author.split()[-1] if ' ' in first_author else first_author
        search_term = f"{search_term} AND {last_name}[Author]"
    
    params = {
        'db': 'pubmed',
        'term': search_term,
        'retmode': 'json',
        'retmax': 3
    }
    
    try:
        response = requests.get(search_url, params=params)
        data = response.json()
        
        if 'esearchresult' in data and 'idlist' in data['esearchresult'] and data['esearchresult']['idlist']:
            pmid = data['esearchresult']['idlist'][0]
            
            # Get article details
            fetch_url = f"{base_url}esummary.fcgi"
            fetch_params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'json'
            }
            
            fetch_response = requests.get(fetch_url, params=fetch_params)
            fetch_data = fetch_response.json()
            
            if 'result' in fetch_data and pmid in fetch_data['result']:
                article = fetch_data['result'][pmid]
                retrieved_title = article.get('title', '')
                
                # Check if the title matches closely
                score = SequenceMatcher(None, title.lower(), retrieved_title.lower()).ratio()
                
                if score > 0.65:  # Reduced threshold for better matching
                    return {
                        'pmid': pmid,
                        'match_score': score,
                        'matched_title': retrieved_title
                    }
    
    except Exception as e:
        print(f"Error searching PubMed: {e}")
    
    return None

def process_pdf_directory(directory_path):
    """Process all PDFs in a directory and write results to CSV"""
    results = []
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    
    for i, pdf_file in enumerate(pdf_files):
        print(f"Processing {i+1}/{len(pdf_files)}: {pdf_file}")
        pdf_path = os.path.join(directory_path, pdf_file)
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        
        # Extract paper info
        papers = extract_paper_info(text)
        print(f"  Found {len(papers)} potential papers in this PDF")
        
        for j, paper in enumerate(papers):
            print(f"  Processing paper {j+1}/{len(papers)}: {paper['title'][:50]}...")
            
            # Give APIs time to breathe (avoid rate limiting)
            time.sleep(1)
            
            # Try CrossRef first
            crossref_result = search_crossref_by_title(paper['title'])
            
            # If no result from CrossRef, try PubMed with title and author
            pubmed_result = None
            if not crossref_result or crossref_result['match_score'] < 0.8:
                time.sleep(1)  # Another pause to avoid rate limits
                pubmed_result = search_pubmed_by_title_and_author(paper['title'], paper.get('authors'))
            
            # Add to results
            result = {
                'file': pdf_file,
                'title': paper['title'],
                'authors': paper.get('authors', ''),
                'year': paper.get('year', ''),
                'journal_info': paper.get('journal_info', ''),
                'doi': crossref_result['doi'] if crossref_result else '',
                'pmid': pubmed_result['pmid'] if pubmed_result else '',
                'match_confidence': crossref_result['match_score'] if crossref_result else (
                    pubmed_result['match_score'] if pubmed_result else 0
                )
            }
            
            results.append(result)
    
    # Write to CSV
    output_file = "extracted_papers.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file', 'title', 'authors', 'year', 'journal_info', 'doi', 'pmid', 'match_confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\nExtracted {len(results)} papers. Data saved to {output_file}")

def main():
    directory = "/Users/cex/OneDrive - University of Edinburgh/Biology/4th Year/Ecology Honours/Dissertation/Code/Scrubbed papers/Archive"
    
    if not os.path.isdir(directory):
        print("Invalid directory path.")
        return
    
    process_pdf_directory(directory)

if __name__ == "__main__":
    main()