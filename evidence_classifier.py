from urllib.parse import urlparse
import re
from models import Evidence

HIGH = {'reuters.com','apnews.com','sec.gov','fda.gov','cdc.gov','nih.gov','who.int','whitehouse.gov','commerce.gov','congress.gov','federalregister.gov','nature.com','science.org'}
MEDIUM = {'bbc.com','bloomberg.com','ft.com','wsj.com','nytimes.com','washingtonpost.com','theguardian.com','cnbc.com','yahoo.com'}
LOW = {'youtube.com','reddit.com','facebook.com','instagram.com','tiktok.com','wikipedia.org'}

def normalized_domain(url_or_domain: str) -> str:
    value = url_or_domain or ''
    domain = urlparse(value).netloc if '://' in value else value
    return domain.lower().removeprefix('www.').split(':')[0]

def authority(domain: str) -> str:
    d = normalized_domain(domain)
    if d.endswith('.gov') or d.endswith('.edu') or any(d == x or d.endswith('.' + x) for x in HIGH): return 'HIGH'
    if any(d == x or d.endswith('.' + x) for x in MEDIUM): return 'MEDIUM'
    if any(d == x or d.endswith('.' + x) for x in LOW): return 'LOW'
    return 'UNKNOWN'

def relationship(claim: str, title: str, snippet: str) -> str:
    tokens = set(re.findall(r'[a-z]{4,}', claim.lower()))
    text = (title + ' ' + snippet).lower()
    overlap = sum(t in text for t in tokens) / max(1, len(tokens))
    contradiction = any(p in text for p in ['no evidence','debunked','false claim','contradicts','did not'])
    if contradiction and overlap >= .35: return 'CONTRADICTION'
    if overlap >= .70: return 'DIRECT_SUPPORT'
    if overlap >= .50: return 'PARTIAL_SUPPORT'
    if overlap >= .30: return 'RELATED_CONTEXT'
    return 'WEAK_EVIDENCE'

def quality_score(authority_tier: str, rel: str, search_score: float) -> float:
    a = {'HIGH':1,'MEDIUM':.7,'LOW':.25,'UNKNOWN':.45}[authority_tier]
    r = {'DIRECT_SUPPORT':1,'PARTIAL_SUPPORT':.75,'RELATED_CONTEXT':.4,'CONTRADICTION':.8,'WEAK_EVIDENCE':.15,'NOISE_SOURCE':0}[rel]
    return round(min(1, .4*a + .4*r + .2*max(0,min(1,search_score))), 3)
