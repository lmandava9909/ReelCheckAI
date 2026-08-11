import json, os, re, shutil, tempfile, time, uuid
from pathlib import Path
from urllib.parse import urlparse
import instaloader, requests
from databricks.sdk import WorkspaceClient
from pydantic import ValidationError
from models import AnalysisResult, Claim, Evidence, ClaimVerdict, ProcessingMetrics
from transcript_correction import TranscriptCorrector
from evidence_classifier import authority, relationship, quality_score, normalized_domain

class InstagramProcessor:
    def __init__(self):
        self.loader = instaloader.Instaloader(download_videos=True, download_video_thumbnails=True, download_comments=False, save_metadata=False, compress_json=False, post_metadata_txt_pattern='')
    def inspect(self, url):
        m = re.search(r'instagram\.com/(reel|p|tv)/([^/]+)/', url)
        if not m: raise ValueError('Use a public Instagram /reel/, /p/, or /tv/ URL.')
        kind, shortcode = m.groups()
        return {'shortcode':shortcode,'post_type':{'reel':'Reel','p':'Post or Carousel','tv':'Video'}[kind]}
    def process(self, url):
        info = self.inspect(url); work = Path(tempfile.mkdtemp(prefix='reelcheck_'))
        post = instaloader.Post.from_shortcode(self.loader.context, info['shortcode'])
        assets=[]
        nodes=list(post.get_sidecar_nodes()) if post.typename == 'GraphSidecar' else [post]
        for i,node in enumerate(nodes):
            is_video=getattr(node,'is_video',False); media_url=getattr(node,'video_url',None) if is_video else getattr(node,'display_url',None)
            if not media_url: continue
            suffix='.mp4' if is_video else '.jpg'; path=work/f'asset_{i:02d}{suffix}'
            with requests.get(media_url,stream=True,timeout=45) as response:
                response.raise_for_status()
                with open(path,'wb') as handle:
                    for chunk in response.iter_content(1024*1024):
                        if chunk: handle.write(chunk)
            assets.append({'index':i,'type':'video' if is_video else 'image','path':str(path)})
        return {**info,'caption':post.caption or '','title':post.title or '' if hasattr(post,'title') else '', 'assets':assets,'workdir':str(work)}

class VisualExtractor:
    def process(self, assets):
        visible=[]
        try:
            import pytesseract
            from PIL import Image
            for asset in assets:
                if asset['type']=='image':
                    try:
                        text=pytesseract.image_to_string(Image.open(asset['path'])).strip()
                        if text: visible.append(f"Image {asset['index']+1}: {text}")
                    except Exception: pass
        except Exception: pass
        return '\n'.join(visible)

class TranscriptExtractor:
    def process(self, assets):
        import whisper
        video=next((a['path'] for a in assets if a['type']=='video'),None)
        if not video: return ''
        return (whisper.load_model(os.getenv('WHISPER_MODEL','base')).transcribe(video,fp16=False).get('text') or '').strip()

class DatabricksLLM:
    def __init__(self): self.client=WorkspaceClient(); self.endpoint=os.getenv('DATABRICKS_LLM_ENDPOINT','databricks-meta-llama-3-1-70b-instruct')
    def json_chat(self,prompt,max_tokens=1600):
        response=self.client.serving_endpoints.query(name=self.endpoint,messages=[{'role':'user','content':prompt}],max_tokens=max_tokens,temperature=.1)
        text=response.choices[0].message.content
        match=re.search(r'```(?:json)?\s*(\{.*\})\s*```',text,re.S) or re.search(r'(\{.*\})',text,re.S)
        if not match: raise ValueError('Model did not return JSON')
        return json.loads(match.group(1))
    def claims(self, content):
        prompt='''You are a conservative social-content analyst. Separate viewpoints, evidence-backed statements, factual claims, opinions, predictions, promotions, and filler. Return JSON only and never invent a claim.\nReturn {"category":"...","viewpoints":["..."],"core_claim":"...","claims":[{"text":"...","claim_type":"...","risk_domain":"Policy|Finance|Health|Legal|Safety|Science|Technology|Consumer|General","priority":"high|medium|low","source_excerpt":"...","quality_score":0.0,"needs_review":false,"review_flags":[]}]}\nCONTENT:\n''' + content[:14000]
        data=self.json_chat(prompt); claims=[]
        for i,item in enumerate(data.get('claims',[])[:7],1): claims.append(Claim(claim_id=f'claim-{i:03d}',**item))
        return data.get('category','General'),data.get('viewpoints',[]),data.get('core_claim',''),claims

class EvidenceRetriever:
    def search(self, claim):
        key=os.getenv('TAVILY_API_KEY')
        if not key: return []
        from tavily import TavilyClient
        data=TavilyClient(api_key=key).search(query=f"{claim.text} official source Reuters AP evidence",search_depth='basic',max_results=5,include_answer=False)
        output=[]; seen=set()
        for i,item in enumerate(data.get('results',[]),1):
            url=item.get('url') or ''
            if not url or url in seen: continue
            seen.add(url); title=item.get('title') or 'Untitled'; snippet=item.get('content') or ''; domain=normalized_domain(url); rel=relationship(claim.text,title,snippet); tier=authority(domain); score=float(item.get('score') or 0)
            output.append(Evidence(evidence_id=f'{claim.claim_id}-ev-{i:02d}',claim_id=claim.claim_id,title=title,url=url,domain=domain,snippet=snippet,relationship=rel,authority_tier=tier,search_score=min(1,score),quality_score=quality_score(tier,rel,score)))
        return sorted(output,key=lambda x:x.quality_score,reverse=True)

class HybridVerdict:
    def process(self, claim, evidence):
        direct=sum(e.relationship=='DIRECT_SUPPORT' for e in evidence); partial=sum(e.relationship=='PARTIAL_SUPPORT' for e in evidence); contra=sum(e.relationship=='CONTRADICTION' for e in evidence); high=sum(e.authority_tier=='HIGH' for e in evidence); quality=sum(e.quality_score for e in evidence[:3])/max(1,min(3,len(evidence)))
        if not evidence: verdict,confidence='Insufficient Evidence',.35
        elif contra>direct+partial: verdict,confidence='Unsupported',min(.9,.55+quality*.35)
        elif direct>=1 and high>=1 and quality>=.7: verdict,confidence='Mostly Supported',min(.92,.65+quality*.3)
        elif direct+partial>=2: verdict,confidence='Partially Supported',min(.82,.55+quality*.3)
        elif contra and (direct or partial): verdict,confidence='Mixed Evidence',.65
        else: verdict,confidence='Insufficient Evidence',.5
        risk=35 if verdict in {'Supported','Mostly Supported'} else 50 if verdict in {'Partially Supported','Mixed Evidence'} else 70
        reasoning=f"Reviewed {len(evidence)} sources: {direct} direct, {partial} partial, and {contra} contradictory; top-source quality averaged {quality:.2f}."
        gap='No independent evidence was retrieved.' if not evidence else ('The evidence does not verify every part of the claim.' if verdict!='Supported' else '')
        takeaway='Reasonable to accept with the stated limitations.' if verdict in {'Supported','Mostly Supported'} else 'Treat the claim cautiously and verify before acting.'
        return ClaimVerdict(claim_id=claim.claim_id,claim_text=claim.text,verdict=verdict,confidence=round(confidence,2),risk_score=risk,evidence_quality_score=round(quality,3),supporting_evidence_count=direct+partial,contradicting_evidence_count=contra,reasoning=reasoning,evidence_gap=gap,user_takeaway=takeaway,evidence=evidence)

class ReelCheckPipeline:
    def process(self,url,manual_caption='',manual_transcript=''):
        start=time.time(); metrics=ProcessingMetrics(); analysis_id='analysis-'+uuid.uuid4().hex[:12]; ext=None
        try:
            t=time.time(); ext=InstagramProcessor().process(url); metrics.extraction_duration_sec=round(time.time()-t,2); metrics.stages_completed.append('extraction')
            caption=ext['caption'] or manual_caption; assets=ext['assets']
            t=time.time(); raw=manual_transcript or TranscriptExtractor().process(assets); metrics.transcription_duration_sec=round(time.time()-t,2); metrics.stages_completed.append('transcription' if raw else 'caption_fallback')
            audit=TranscriptCorrector().correct_with_caption_context(raw,caption,ext.get('title',''))
            visible=VisualExtractor().process(assets)
            combined='\n\n'.join(x for x in [f'CAPTION: {caption}',f'TRANSCRIPT: {audit.corrected_transcript}',f'VISIBLE TEXT: {visible}'] if x.split(':',1)[1].strip())
            t=time.time(); category,views,core,claims=DatabricksLLM().claims(combined); metrics.claim_duration_sec=round(time.time()-t,2); metrics.llm_calls+=1; metrics.stages_completed.append('claims')
            t=time.time(); verdicts=[]
            for claim in claims:
                verdicts.append(HybridVerdict().process(claim,EvidenceRetriever().search(claim))); metrics.search_api_calls+=1
            metrics.evidence_duration_sec=round(time.time()-t,2); metrics.stages_completed.append('evidence_and_verdict')
            positive=sum(v.verdict in {'Supported','Mostly Supported'} for v in verdicts)
            overall='Mostly Supported' if verdicts and positive==len(verdicts) else 'Mixed Evidence' if positive else (verdicts[0].verdict if verdicts else 'Insufficient Evidence')
            reliability=round(100*sum(v.confidence*v.evidence_quality_score for v in verdicts)/len(verdicts)) if verdicts else 0
            risk=round(sum(v.risk_score for v in verdicts)/len(verdicts)) if verdicts else 65
            noise=[p for p in ['going ballistic','going insane','follow for more','link in my bio'] if p in combined.lower()]
            return AnalysisResult(analysis_id=analysis_id,url=url.split('?')[0],shortcode=ext['shortcode'],post_type=ext['post_type'],caption=caption,asset_count=len(assets),visual_text=visible,transcript=audit,category=category,viewpoints=views,core_claim=core,claims=claims,verdicts=verdicts,overall_verdict=overall,reliability_score=reliability,risk_score=risk,content_value_score=max(0,min(100,reliability-len(noise)*5+20)),what_matters=[v.claim_text for v in verdicts if v.verdict in {'Supported','Mostly Supported','Partially Supported'}],cautions=[v.evidence_gap for v in verdicts if v.evidence_gap],promotional_noise=noise,recommended_action='Use evidence-backed findings as context. Verify financial, health, legal, and safety decisions using primary sources.',metrics=metrics)
        finally:
            metrics.total_duration_sec=round(time.time()-start,2)
            if ext: shutil.rmtree(ext.get('workdir',''),ignore_errors=True)
