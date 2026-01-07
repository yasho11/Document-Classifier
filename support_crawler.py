import requests
import feedparser
import os
import time
import re
from bs4 import BeautifulSoup

# Create health folder
os.makedirs('health', exist_ok=True)

# Health RSS feeds (BBC + Medical sources)
health_rss_feeds = [
    'http://feeds.bbci.co.uk/news/health/rss.xml',  # BBC Health
    'https://medicalxpress.com/rss-feed/health/',  # Medical Xpress
    'https://www.medicalnewstoday.com/rss/featuredhealth.xml',  # Medical News Today
    'https://www.nih.gov/news-events/news-releases/rss',  # NIH News
    'https://www.who.int/feeds/entity/csr/don/en/rss.xml'  # WHO Disease Outbreaks
]

# User-Agent header to avoid blocking
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Function to clean text
def clean_text(text):
    # Remove extra whitespace, newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove any HTML tags that might remain
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

# Function to extract article text
def extract_article_text(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Try common article selectors
        selectors = [
            'article', 
            '.article-body', 
            '.post-content',
            '#article-body',
            '.story-body',
            '[itemprop="articleBody"]',
            'div.article-content'
        ]
        
        article_text = ""
        for selector in selectors:
            article = soup.select_one(selector)
            if article:
                article_text = article.get_text()
                break
        
        # If no specific selector found, get main content
        if not article_text:
            main = soup.find('main') or soup.find('div', role='main')
            if main:
                article_text = main.get_text()
            else:
                # Fallback: get all paragraphs
                paragraphs = soup.find_all('p')
                article_text = ' '.join([p.get_text() for p in paragraphs if len(p.get_text()) > 50])
        
        return clean_text(article_text[:3000])  # Limit to 3000 chars
    
    except Exception as e:
        print(f"    Error extracting {url}: {e}")
        return ""

# Main crawling function
def crawl_health_articles(target_count=150):
    collected_articles = []
    article_count = 0 
    
    print(f"Starting to collect {target_count} health articles...")
    print("=" * 50)
    
    for rss_url in health_rss_feeds:
        if article_count >= target_count:
            break
            
        print(f"\nFetching from: {rss_url}")
        
        try:
            # Parse RSS feed
            feed = feedparser.parse(rss_url)
            print(f"  Found {len(feed.entries)} articles in feed")
            
            for entry in feed.entries:
                if article_count >= target_count:
                    break
                
                print(f"  Processing: {entry.title[:60]}...")
                
                # Try to get content
                article_content = ""
                
                # First try RSS content if available
                if hasattr(entry, 'content'):
                    for content in entry.content:
                        article_content = content.value
                        break
                
                # If no RSS content, crawl the article page
                if not article_content and hasattr(entry, 'link'):
                    article_content = extract_article_text(entry.link)
                    time.sleep(0.5)  # Be polite
                
                # If we have content, save it
                if article_content and len(article_content) > 200:
                    # Save to file
                    filename = f"health/health_{article_count+1:03d}.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(article_content)
                    
                    collected_articles.append({
                        'title': entry.title if hasattr(entry, 'title') else 'No title',
                        'source': rss_url,
                        'filename': filename
                    })
                    
                    article_count += 1
                    print(f"    ✓ Saved as {filename} ({len(article_content)} chars)")
                    
                    # Optional: Save every 10 articles
                    if article_count % 10 == 0:
                        print(f"\nProgress: {article_count}/{target_count} articles collected")
                
        except Exception as e:
            print(f"  Error with RSS feed {rss_url}: {e}")
            continue
    
    return collected_articles, article_count

# Run the crawler
if __name__ == "__main__":
    print("HEALTH NEWS CRAWLER")
    print("=" * 50)
    
    try:
        articles, count = crawl_health_articles(target_count=150)
        
        print("\n" + "=" * 50)
        print(f"CRAWLING COMPLETE!")
        print(f"Total articles collected: {count}")
        print(f"Files saved in: {os.path.abspath('health')}")
        
        # Show sample of collected articles
        if articles:
            print("\nSample of collected articles:")
            for i, article in enumerate(articles[:5]):
                print(f"  {i+1}. {article['title'][:70]}...")
        
        # Verify files were created
        files = os.listdir('health')
        print(f"\nFiles in health folder: {len(files)}")
        
        if files:
            # Show first file content preview
            with open(os.path.join('health', files[0]), 'r', encoding='utf-8') as f:
                preview = f.read(200)
                print(f"\nFirst file preview:\n{preview}...")
    
    except KeyboardInterrupt:
        print("\n\nCrawling stopped by user.")
        files = os.listdir('health')
        print(f"Collected {len(files)} articles before stopping.")
    
    print("\nDone! Press Enter to exit...")
    input()