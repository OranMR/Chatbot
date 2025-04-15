import pandas as pd
import requests
from tqdm import tqdm
import os
import re
import time
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# === CONFIGURATION ===
CSV_PATH = "C:/Users/cex/OneDrive - University of Edinburgh/Biology/4th Year/Ecology Honours/Dissertation/Code/Scrubbed papers/recent_pubmed_articles_test.csv"
PROXY_PREFIX = "https://login.eux.idm.oclc.org/login?url="
EMAIL = "s2194841@ed.ac.uk"  

# Directory where downloaded PDFs will be saved
SAVE_DIR = "downloaded_papers"
os.makedirs(SAVE_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("paper_downloads.log"),
        logging.StreamHandler()
    ]
)


def clean_filename(title):
    """Sanitize filename to avoid OS errors."""
    return re.sub(r'[\\/*?:"<>|]', "", title)

def create_session_with_retries():
    """Create a session with retry capability"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
    return session

def create_authenticated_session():
    """Create a session with authentication for institutional access
    
    NOTE: You need to modify this function to add actual authentication
    mechanisms required by your institution
    """
    session = create_session_with_retries()
    
    # CUSTOMIZE THIS PART:
    # For Edinburgh, you might need to implement a proper login flow
    # Example (you'll need to uncomment and modify this):
    login_url = "https://login.eux.idm.oclc.org/login?url="
    credentials = {"username": "s2194841", "password": "clecks82"}
    session.post(login_url, data=credentials)
    
    return session

def test_institutional_access(session):
    """Test if institutional access is working"""
    # Try accessing a known paywalled article
    test_url = f"{PROXY_PREFIX}https://www.sciencedirect.com/science/article/pii/S0169534719300795/pdfft"
    try:
        response = session.get(test_url, timeout=30)
        if response.status_code == 200 and "application/pdf" in response.headers.get("Content-Type", ""):
            logging.info("✅ Institutional access is working!")
            return True
        else:
            logging.warning(f"⚠️ Institutional access test failed with status code: {response.status_code}")
            return False
    except Exception as e:
        logging.error(f"⚠️ Error testing institutional access: {e}")
        return False

def download_with_retry(url, filepath, session, max_retries=3):
    """Download with retries and proper error handling"""
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=30)
            if response.status_code == 200 and "application/pdf" in response.headers.get("Content-Type", ""):
                with open(filepath, "wb") as f:
                    f.write(response.content)
                return True
            elif response.status_code == 429:  # Rate limit
                wait_time = min(2 ** attempt, 60)  # Exponential backoff
                logging.info(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.warning(f"Failed with status code: {response.status_code}")
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout on attempt {attempt+1}")
        except Exception as e:
            logging.error(f"Error on attempt {attempt+1}: {e}")
        
        # Wait before retrying
        if attempt < max_retries - 1:
            time.sleep(min(2 ** attempt, 10))
    
    return False

def try_unpaywall(doi, title, session):
    """Try downloading from Unpaywall"""
    filepath = os.path.join(SAVE_DIR, clean_filename(title)[:100] + ".pdf")
    
    try:
        unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email={EMAIL}"
        resp = session.get(unpaywall_url)
        if resp.status_code == 200:
            data = resp.json()
            oa_location = data.get("best_oa_location")
            if oa_location and oa_location.get("url_for_pdf"):
                pdf_url_unpaywall = oa_location["url_for_pdf"]
                logging.info(f"🔓 Open Access PDF found via Unpaywall: {pdf_url_unpaywall}")
                return download_with_retry(pdf_url_unpaywall, filepath, session)
    except Exception as e:
        logging.error(f"Unpaywall error for {doi}: {e}")
    
    return False

def try_manual_url(doi, title, pdf_url, session):
    """Try downloading from a manually provided URL"""
    if not pdf_url:
        return False
        
    filepath = os.path.join(SAVE_DIR, clean_filename(title)[:100] + ".pdf")
    
    try:
        logging.info(f"⚙️ Trying manual PDF link for {doi}: {pdf_url}")
        return download_with_retry(pdf_url, filepath, session)
    except Exception as e:
        logging.error(f"Error using manual PDF link for {doi}: {e}")
    
    return False

def try_doi_direct(doi, title, session):
    """Try direct DOI resolution"""
    filepath = os.path.join(SAVE_DIR, clean_filename(title)[:100] + ".pdf")
    
    try:
        doi_url = f"https://doi.org/{doi}"
        logging.info(f"🔍 Trying direct DOI resolution: {doi_url}")
        
        # First follow the DOI to find the actual publisher page
        response = session.get(doi_url, allow_redirects=True)
        if response.status_code == 200:
            # Some publishers might have the PDF link on this page
            if "application/pdf" in response.headers.get("Content-Type", ""):
                with open(filepath, "wb") as f:
                    f.write(response.content)
                return True
    except Exception as e:
        logging.error(f"DOI direct error for {doi}: {e}")
    
    return False

def download_with_institutional_access(doi, title, session):
    """Try downloading through institutional proxy"""
    filepath = os.path.join(SAVE_DIR, clean_filename(title)[:100] + ".pdf")
    
    if doi:
        proxy_url = f"{PROXY_PREFIX}https://doi.org/{doi}"
        logging.info(f"🏛️ Trying institutional access: {proxy_url}")
        
        try:
            return download_with_retry(proxy_url, filepath, session)
        except Exception as e:
            logging.error(f"Institutional access error: {e}")
    
    return False

def try_publisher_specific(doi, title, session):
    """Handle specific publishers based on DOI patterns"""
    filepath = os.path.join(SAVE_DIR, clean_filename(title)[:100] + ".pdf")
    
    # Handle Wiley
    if "10.1111" in doi or "10.1002" in doi:
        wiley_url = f"{PROXY_PREFIX}https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}"
        logging.info(f"📚 Trying Wiley-specific approach: {wiley_url}")
        try:
            return download_with_retry(wiley_url, filepath, session)
        except Exception as e:
            logging.error(f"Wiley error: {e}")
    
    # Handle Springer
    if "10.1007" in doi:
        springer_url = f"{PROXY_PREFIX}https://link.springer.com/content/pdf/{doi}.pdf"
        logging.info(f"📚 Trying Springer-specific approach: {springer_url}")
        try:
            return download_with_retry(springer_url, filepath, session)
        except Exception as e:
            logging.error(f"Springer error: {e}")
    
    # Handle Elsevier/ScienceDirect
    if "10.1016" in doi:
        # Extract the PII from the DOI if possible
        pii = doi.split('/')[-1]
        science_direct_url = f"{PROXY_PREFIX}https://www.sciencedirect.com/science/article/pii/{pii}/pdfft"
        logging.info(f"📚 Trying ScienceDirect-specific approach: {science_direct_url}")
        try:
            return download_with_retry(science_direct_url, filepath, session)
        except Exception as e:
            logging.error(f"ScienceDirect error: {e}")
    
    # Handle Oxford Academic
    if "10.1093" in doi:
        oxford_url = f"{PROXY_PREFIX}https://academic.oup.com/view-large/doi/{doi}/pdf"
        logging.info(f"📚 Trying Oxford Academic-specific approach: {oxford_url}")
        try:
            return download_with_retry(oxford_url, filepath, session)
        except Exception as e:
            logging.error(f"Oxford error: {e}")
    
    # Handle Taylor & Francis
    if "10.1080" in doi:
        tf_url = f"{PROXY_PREFIX}https://www.tandfonline.com/doi/pdf/{doi}"
        logging.info(f"📚 Trying Taylor & Francis-specific approach: {tf_url}")
        try:
            return download_with_retry(tf_url, filepath, session)
        except Exception as e:
            logging.error(f"Taylor & Francis error: {e}")
    
    # Handle SAGE
    if "10.1177" in doi:
        sage_url = f"{PROXY_PREFIX}https://journals.sagepub.com/doi/pdf/{doi}"
        logging.info(f"📚 Trying SAGE-specific approach: {sage_url}")
        try:
            return download_with_retry(sage_url, filepath, session)
        except Exception as e:
            logging.error(f"SAGE error: {e}")
            
    # Handle Royal Society
    if "10.1098" in doi:
        rs_url = f"{PROXY_PREFIX}https://royalsocietypublishing.org/doi/pdf/{doi}"
        logging.info(f"📚 Trying Royal Society-specific approach: {rs_url}")
        try:
            return download_with_retry(rs_url, filepath, session)
        except Exception as e:
            logging.error(f"Royal Society error: {e}")
    
    return False

def try_other_repositories(doi, title, session):
    """Try other repositories like arXiv, PMC, etc."""
    filepath = os.path.join(SAVE_DIR, clean_filename(title)[:100] + ".pdf")
    
    # Try arXiv (if you can extract arXiv ID)
    arxiv_pattern = r"arXiv:(\d+\.\d+)"
    match = re.search(arxiv_pattern, title)
    if match:
        arxiv_id = match.group(1)
        arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        logging.info(f"📑 Trying arXiv: {arxiv_url}")
        try:
            return download_with_retry(arxiv_url, filepath, session)
        except Exception as e:
            logging.error(f"arXiv error: {e}")
    
    # Try PubMed Central 
    # First check if it's a PMC article and extract the PMCID
    pmc_pattern = r"PMC(\d+)"
    match = re.search(pmc_pattern, title)
    if match:
        pmcid = match.group(1)
        pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/pdf/"
        logging.info(f"📑 Trying PubMed Central: {pmc_url}")
        try:
            return download_with_retry(pmc_url, filepath, session)
        except Exception as e:
            logging.error(f"PMC error: {e}")
    
    # Try bioRxiv/medRxiv - if title or paper mentions it
    if "biorxiv" in title.lower() or "medrxiv" in title.lower():
        # Try to extract the bioRxiv ID format (typically like 10.1101/2020.01.01.123456)
        biorxiv_pattern = r"10\.1101/(\d+\.\d+\.\d+\.\d+)"
        match = re.search(biorxiv_pattern, doi if doi else "")
        if match:
            biorxiv_id = match.group(1)
            biorxiv_url = f"https://www.biorxiv.org/content/10.1101/{biorxiv_id}.full.pdf"
            logging.info(f"📑 Trying bioRxiv: {biorxiv_url}")
            try:
                return download_with_retry(biorxiv_url, filepath, session)
            except Exception as e:
                logging.error(f"bioRxiv error: {e}")
    
    return False

def download_pdf(doi, title, pdf_url=None):
    """Main function to download a PDF using multiple methods"""
    # Create a standard session for most requests
    session = create_session_with_retries()
    
    # Create an authenticated session for institutional access
    auth_session = create_authenticated_session()
    
    filename = clean_filename(title)[:100] + ".pdf"
    filepath = os.path.join(SAVE_DIR, filename)
    
    # Check if file already exists
    if os.path.exists(filepath):
        logging.info(f"✅ File already exists: {filename}")
        return True

    logging.info(f"⬇️ Attempting to download: {title} (DOI: {doi})")
    
    # Try different methods in sequence
    
    # 1. Try Unpaywall
    if try_unpaywall(doi, title, session):
        logging.info(f"✅ Downloaded via Unpaywall: {title}")
        return True
    
    # 2. Try manual override URL
    if try_manual_url(doi, title, pdf_url, session):
        logging.info(f"✅ Downloaded via manual URL: {title}")
        return True
    
    # 3. Try direct DOI resolution
    if try_doi_direct(doi, title, session):
        logging.info(f"✅ Downloaded via direct DOI: {title}")
        return True
    
    # 4. Try institutional access
    if download_with_institutional_access(doi, title, auth_session):
        logging.info(f"✅ Downloaded via institutional access: {title}")
        return True
    
    # 5. Try publisher-specific approaches
    if try_publisher_specific(doi, title, auth_session):
        logging.info(f"✅ Downloaded via publisher-specific approach: {title}")
        return True
    
    # 6. Try other repositories
    if try_other_repositories(doi, title, session):
        logging.info(f"✅ Downloaded via other repositories: {title}")
        return True

    logging.warning(f"❌ Could not download: {doi} - {title}")
    return False

def download_papers_from_csv(csv_path):
    """Download papers from a CSV file"""
    df = pd.read_csv(csv_path)
    results = {"success": 0, "failed": 0, "skipped": 0}
    
    # Create a list to store failed downloads for reporting
    failed_papers = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading papers"):
        title = str(row.get("title", "untitled"))
        doi = row.get("doi")
        pdf_url = row.get("pdf_url") if "pdf_url" in row else None

        if pd.isna(doi):
            logging.info(f"Skipping paper without DOI: {title}")
            results["skipped"] += 1
            continue
            
        success = download_pdf(doi, title, pdf_url)
        if success:
            results["success"] += 1
        else:
            results["failed"] += 1
            failed_papers.append({"title": title, "doi": doi})
    
    # Print summary
    logging.info(f"✅ Download summary: {results['success']} successful, {results['failed']} failed, {results['skipped']} skipped")
    
    # Print failed papers
    if failed_papers:
        logging.info("Failed downloads:")
        for paper in failed_papers:
            logging.info(f"  - {paper['title']} (DOI: {paper['doi']})")
    
    return results

# === Run the script ===
if __name__ == "__main__":
    download_papers_from_csv(CSV_PATH)
    logging.info("✅ Script completed!")