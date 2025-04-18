import pandas as pd
from Bio import Entrez
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from tqdm import tqdm
import time
from xml.sax.saxutils import escape
import os
import re

# Set your email for Entrez API use
Entrez.email = "s2194841@ed.ac.uk"  

def fetch_pubmed_metadata(pmid):
    """Fetch metadata and abstract from PubMed for a given PMID."""
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="xml")
        records = Entrez.read(handle)
        article = records["PubmedArticle"][0]["MedlineCitation"]["Article"]

        title = article.get("ArticleTitle", "No Title Available")
        abstract = article.get("Abstract", {}).get("AbstractText", ["No Abstract Available"])[0]
        authors = article.get("AuthorList", [])
        author_names = ", ".join(
            [f"{a.get('ForeName', '')} {a.get('LastName', '')}".strip() for a in authors]
        )
        journal = article.get("Journal", {}).get("Title", "No Journal Info")
        pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        date_parts = [pub_date.get(part, '') for part in ["Year", "Month", "Day"]]
        date_str = " ".join(filter(None, date_parts)) or "No Date"

        return {
            "Title": title,
            "Authors": author_names,
            "Journal": journal,
            "Date": date_str,
            "Abstract": abstract,
            "PMID": pmid
        }
    except Exception as e:
        print(f"Error fetching PMID {pmid}: {e}")
        return {
            "Title": "Error",
            "Authors": "",
            "Journal": "",
            "Date": "",
            "Abstract": "Could not retrieve abstract.",
            "PMID": pmid
        }
    
def sanitize_filename(name):
    name = name.replace("&", "and")  # replace ampersand
    name = re.sub(r'[\/\\\*\<\>\|\:\\"\?]', '', name)  # remove illegal characters
    return name.strip()

def create_single_pdf(data, output_dir="abstract_pdfs_test"):
    """Generate a single PDF file for one article."""
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit title length and sanitize
    sanitized_title = sanitize_filename(data['Title'])
    shortened_title = sanitized_title[:100]
    
    # Process authors according to the naming scheme
    author_list = []
    if data['Authors']:
        author_list = [author.strip() for author in data['Authors'].split(',') if author.strip()]
    
    if not author_list:
        author_part = "Unknown"
    elif len(author_list) == 1:
        name_parts = author_list[0].split()
        surname = name_parts[-1] if name_parts else "Unknown"
        author_part = surname
    elif len(author_list) == 2:
        author1_parts = author_list[0].split()
        author2_parts = author_list[1].split()
        surname1 = author1_parts[-1] if author1_parts else "Unknown"
        surname2 = author2_parts[-1] if author2_parts else "Unknown"
        author_part = f"{surname1} & {surname2}"
    else:
        first_author_parts = author_list[0].split()
        surname = first_author_parts[-1] if first_author_parts else "Unknown"
        author_part = f"{surname} et al"
        author_part = sanitize_filename(author_part)
    
    filename = f"{shortened_title}-{author_part}"

    if len(filename) > 130:
        filename = f"{shortened_title[:94]}... {author_part[27]}"
    
    # Add .pdf extension separately
    output_filename = f"{filename}.pdf"
    output_path = os.path.join(output_dir, output_filename)
    
    # Create the PDF
    doc = SimpleDocTemplate(output_path, pagesize=LETTER)
    styles = getSampleStyleSheet()
    story = []

    # Add content to PDF
    story.append(Paragraph(f"<b>{escape(data['Title'])} {escape(data['Date'])}</b>", styles['Heading2']))
    story.append(Paragraph(f"<b>Authors:</b> {escape(data['Authors'])}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Abstract:</b>", styles['Normal']))
    story.append(Paragraph(escape(data['Abstract']), styles['Normal']))

    doc.build(story)
    return output_path

def main():
    csv_file = "C:/Users/cex/OneDrive - University of Edinburgh/Biology/4th Year/Ecology Honours/Dissertation/Code/Scrubbed papers/PMIDs test.csv" 
    df = pd.read_csv(csv_file)

    if "pmid" not in df.columns:
        print("❌ CSV file must contain a 'pmid' column.")
        return

    pmid_list = df["pmid"].dropna().astype(str).tolist()
    output_dir = "abstract_pdfs_test"
     
    print("🔍 Fetching PubMed records and creating PDFs...")
    for pmid in tqdm(pmid_list, desc="Processing abstracts"):
        data = fetch_pubmed_metadata(pmid)
        pdf_path = create_single_pdf(data, output_dir)
        print(f"✅ Created PDF: {os.path.basename(pdf_path)}")
        time.sleep(0)  # Small delay to be nice to NCBI servers

    print(f"📚 All PDFs have been saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    main()