import pandas as pd
from Bio import Entrez
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from tqdm import tqdm
import time
from xml.sax.saxutils import escape


# Set your email for Entrez API use
Entrez.email = "s2194841@ed.ac.uk"  # Replace with your actual email

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
            "Abstract": abstract
        }

    except Exception as e:
        print(f"Error fetching PMID {pmid}: {e}")
        return {
            "Title": "Error",
            "Authors": "",
            "Journal": "",
            "Date": "",
            "Abstract": "Could not retrieve abstract."
        }

def create_pdf(data_list, output_path="pubmed_abstracts.pdf"):
    """Generate a PDF from the list of article metadata."""
    doc = SimpleDocTemplate(output_path, pagesize=LETTER)
    styles = getSampleStyleSheet()
    story = []

    for data in data_list:
        story.append(Paragraph(f"<b>Title:</b> {escape(data['Title'])}", styles['Heading2']))
        story.append(Paragraph(f"<b>Authors:</b> {escape(data['Authors'])}", styles['Normal']))
        story.append(Paragraph(f"<b>Journal:</b> {escape(data['Journal'])}", styles['Normal']))
        story.append(Paragraph(f"<b>Date:</b> {escape(data['Date'])}", styles['Normal']))
        story.append(Spacer(1, 6))
        story.append(Paragraph("<b>Abstract:</b>", styles['Heading3']))
        story.append(Paragraph(escape(data['Abstract']), styles['Normal']))
        story.append(Spacer(1, 20))

    doc.build(story)
    print(f"✅ PDF saved to {output_path}")


def main():
    csv_file = "PMIDs.csv"  
    df = pd.read_csv(csv_file)

    if "pmid" not in df.columns:
        print("❌ CSV file must contain a 'pmid' column.")
        return

    pmid_list = df["pmid"].dropna().astype(str).tolist()
    results = []

    print("🔍 Fetching PubMed records...")
    for pmid in tqdm(pmid_list, desc="Fetching abstracts"):
        data = fetch_pubmed_metadata(pmid)
        results.append(data)
        time.sleep(0)  

    print("📝 Generating PDF...")
    create_pdf(results)

if __name__ == "__main__":
    main()
