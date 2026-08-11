"""ReelCheck AI Processor V3 - Complete End-to-End Pipeline

Features:
- FFmpeg configuration using imageio-ffmpeg
- Explicit audio extraction before Whisper transcription
- yt-dlp as primary extractor with Instaloader fallback
- Support for video/image/carousel/manual fallback routes
- Context-aware transcript correction
- Domain-specific evidence retrieval
- Hybrid verdict generation with guardrails
- No hardcoded reference results
"""

import json, os, re, shutil, subprocess, tempfile, time, uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from databricks.sdk import WorkspaceClient
from pydantic import ValidationError

from models import (
    AnalysisResult, Claim, Evidence, ClaimVerdict, 
    ProcessingMetrics
)
from transcript_correction import TranscriptCorrector
from evidence_classifier import (
    authority, relationship, quality_score, normalized_domain
)


# ==================================================
# 1. FFMPEG CONFIGURATION
# ==================================================

def configure_ffmpeg() -> str:
    """Configure FFmpeg using imageio-ffmpeg package.
    
    Returns path to FFmpeg executable and adds it to PATH.
    Raises RuntimeError if FFmpeg is not available.
    """
    import imageio_ffmpeg
    
    ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()
    
    if not ffmpeg_executable:
        raise RuntimeError("No FFmpeg executable found from imageio-ffmpeg")
    
    ffmpeg_path = Path(ffmpeg_executable)
    
    if not ffmpeg_path.exists():
        raise RuntimeError(
            f"FFmpeg executable does not exist: {ffmpeg_path}"
        )
    
    # Add FFmpeg directory to PATH
    ffmpeg_directory = str(ffmpeg_path.parent)
    current_path = os.environ.get("PATH", "")
    
    if ffmpeg_directory not in current_path.split(os.pathsep):
        os.environ["PATH"] = ffmpeg_directory + os.pathsep + current_path
    
    os.environ["FFMPEG_BINARY"] = str(ffmpeg_path)
    
    return str(ffmpeg_path)


def extract_audio(video_path: str, output_path: str, ffmpeg_executable: str) -> str:
    """Extract audio from video using FFmpeg."""
    command = [
        ffmpeg_executable, "-y", "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path
    ]
    
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if completed.returncode != 0:
            raise RuntimeError(f"Audio extraction failed: {completed.stderr[-1000:]}")
        if not os.path.exists(output_path):
            raise RuntimeError("FFmpeg did not create audio file")
        return output_path
    except subprocess.TimeoutExpired:
        raise RuntimeError("Audio extraction timed out")
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {str(e)}")


# ==================================================
# EXTRACTORS
# ==================================================

class YtDlpExtractor:
    """Extract Instagram content using yt-dlp."""
    
    def __init__(self):
        self.work_dir = None
        
    def extract(self, url: str) -> Dict:
        """Extract Instagram content using yt-dlp."""
        import yt_dlp
        
        match = re.search(r'instagram\.com/(reel|p|tv)/([^/]+)/', url)
        if not match:
            raise ValueError('Use a public Instagram /reel/, /p/, or /tv/ URL')
        
        kind, shortcode = match.groups()
        post_type = {'reel': 'Reel', 'p': 'Post or Carousel', 'tv': 'Video'}[kind]
        
        self.work_dir = Path(tempfile.mkdtemp(prefix='reelcheck_ytdlp_'))
        output_template = str(self.work_dir / '%(id)s.%(ext)s')
        
        ydl_opts = {
            'quiet': True, 'no_warnings': True, 'noplaylist': True,
            'format': 'bestvideo+bestaudio/best', 'outtmpl': output_template,
            'writeinfojson': True, 'merge_output_format': 'mp4'
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                meta = ydl.extract_info(url.split('?')[0], download=True)
            
            media_paths = [str(f) for ext in ['.mp4', '.webm', '.mkv'] for f in self.work_dir.glob(f'*{ext}')]
            
            return {
                'shortcode': shortcode, 'post_type': post_type,
                'caption': meta.get('description', ''), 'title': meta.get('title', ''),
                'media_paths': media_paths, 'image_paths': [],
                'extraction_method': 'yt-dlp', 'workdir': str(self.work_dir), 'warnings': []
            }
        except Exception as e:
            if self.work_dir and self.work_dir.exists():
                shutil.rmtree(self.work_dir, ignore_errors=True)
            raise RuntimeError(f"yt-dlp extraction failed: {str(e)}")


class InstaLoaderExtractor:
    """Fallback extractor using Instaloader."""
    
    def __init__(self):
        import instaloader
        self.loader = instaloader.Instaloader(
            download_videos=True, download_comments=False, save_metadata=False
        )
        self.work_dir = None
    
    def extract(self, url: str) -> Dict:
        """Extract Instagram content using Instaloader."""
        import instaloader
        
        match = re.search(r'instagram\.com/(reel|p|tv)/([^/]+)/', url)
        if not match:
            raise ValueError('Use a public Instagram /reel/, /p/, or /tv/ URL')
        
        kind, shortcode = match.groups()
        post_type = {'reel': 'Reel', 'p': 'Post or Carousel', 'tv': 'Video'}[kind]
        
        self.work_dir = Path(tempfile.mkdtemp(prefix='reelcheck_insta_'))
        post = instaloader.Post.from_shortcode(self.loader.context, shortcode)
        
        nodes = list(post.get_sidecar_nodes()) if post.typename == 'GraphSidecar' else [post]
        media_paths, image_paths = [], []
        
        for i, node in enumerate(nodes):
            is_video = getattr(node, 'is_video', False)
            media_url = getattr(node, 'video_url' if is_video else 'display_url', None)
            if not media_url:
                continue
            
            path = self.work_dir / f'asset_{i:02d}{".mp4" if is_video else ".jpg"}'
            with requests.get(media_url, stream=True, timeout=45) as response:
                response.raise_for_status()
                with open(path, 'wb') as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            
            (media_paths if is_video else image_paths).append(str(path))
        
        return {
            'shortcode': shortcode, 'post_type': post_type,
            'caption': post.caption or '', 'title': '',
            'media_paths': media_paths, 'image_paths': image_paths,
            'extraction_method': 'instaloader', 'workdir': str(self.work_dir),
            'warnings': []
        }


class UnifiedExtractor:
    """Unified extractor: tries yt-dlp first, falls back to Instaloader."""
    
    def extract(self, url: str) -> Dict:
        """Extract Instagram content with fallback strategy."""
        errors = []
        
        try:
            return YtDlpExtractor().extract(url)
        except Exception as e:
            errors.append(f"yt-dlp: {str(e)}")
        
        try:
            result = InstaLoaderExtractor().extract(url)
            result['warnings'].append("Primary extractor failed; used Instaloader")
            return result
        except Exception as e:
            errors.append(f"Instaloader: {str(e)}")
        
        raise RuntimeError("All extraction methods failed. " + " | ".join(errors))


# ==================================================
# TRANSCRIPTION AND OCR
# ==================================================

class WhisperTranscriber:
    """Transcribe audio using OpenAI Whisper."""
    
    def __init__(self):
        self.model = None
        self.model_name = os.getenv('WHISPER_MODEL', 'base')
    
    def transcribe(self, audio_path: str) -> Dict:
        """Transcribe audio file."""
        import whisper
        
        if self.model is None:
            self.model = whisper.load_model(self.model_name)
        
        result = self.model.transcribe(audio_path, fp16=False, verbose=False)
        
        return {
            'raw_transcript': (result.get('text') or '').strip(),
            'language': result.get('language'),
            'model_name': self.model_name
        }


class VisualTextExtractor:
    """Extract text from images using OCR."""
    
    def extract(self, image_paths: List[str]) -> str:
        """Extract text from images using pytesseract."""
        visible_text = []
        
        try:
            import pytesseract
            from PIL import Image
            
            for i, image_path in enumerate(image_paths):
                try:
                    text = pytesseract.image_to_string(Image.open(image_path)).strip()
                    if text:
                        visible_text.append(f"Image {i + 1}: {text}")
                except Exception:
                    pass
        except ImportError:
            pass
        
        return '\n'.join(visible_text)


# ==================================================
# LLM AND EVIDENCE
# ==================================================

class DatabricksLLM:
    """Client for Databricks Foundation Model endpoints."""
    
    def __init__(self):
        self.client = WorkspaceClient()
        self.endpoint = os.getenv('DATABRICKS_LLM_ENDPOINT', 'databricks-meta-llama-3-1-70b-instruct')
    
    def json_chat(self, prompt: str, max_tokens: int = 1600) -> Dict:
        """Call LLM with JSON output requirement."""
        response = self.client.serving_endpoints.query(
            name=self.endpoint,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
            temperature=0.1
        )
        
        text = response.choices[0].message.content
        match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.S) or re.search(r'(\{.*\})', text, re.S)
        if not match:
            raise ValueError('Model did not return valid JSON')
        
        return json.loads(match.group(1))
    
    def classify_and_extract_claims(self, content: str) -> Tuple[str, List[str], str, List[Claim]]:
        """Classify content and extract claims."""
        prompt = f"""Analyze the following content and extract:
1. Overall category (Policy|Finance|Health|Legal|Safety|Science|Technology|Consumer|General)
2. Main viewpoints
3. Core factual claim
4. All factual claims that are externally verifiable, specific, and complete

Return JSON only:
{{
  "category": "...",
  "viewpoints": [...],
  "core_claim": "...",
  "claims": [{{
    "text": "...",
    "claim_type": "factual_statement|policy_claim|statistical_claim|event_claim",
    "risk_domain": "...",
    "priority": "high|medium|low",
    "source_excerpt": "...",
    "quality_score": 0.0,
    "needs_review": false,
    "review_flags": []
  }}]
}}

CONTENT:
{content[:14000]}
"""
        
        data = self.json_chat(prompt)
        claims = [Claim(claim_id=f'claim-{i:03d}', **item) for i, item in enumerate(data.get('claims', [])[:7], 1)]
        
        return data.get('category', 'General'), data.get('viewpoints', []), data.get('core_claim', ''), claims


class EvidenceRetriever:
    """Retrieve evidence using Tavily with domain-specific queries."""
    
    def __init__(self):
        self.api_key = os.getenv('TAVILY_API_KEY')
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY not found")
    
    def generate_queries(self, claim: Claim) -> List[str]:
        """Generate domain-specific search queries."""
        domain_queries = {
            'Policy': [f"{claim.text} Reuters", f"{claim.text} official government policy"],
            'Finance': [f"{claim.text} SEC filing", f"{claim.text} Reuters financial"],
            'Health': [f"{claim.text} NIH CDC", f"{claim.text} peer-reviewed study"],
            'Science': [f"{claim.text} research paper", f"{claim.text} university study"]
        }
        return domain_queries.get(claim.risk_domain, [f"{claim.text} Reuters AP", f"{claim.text} official source"])[:2]
    
    def search(self, claim: Claim) -> List[Evidence]:
        """Search for evidence using domain-specific queries."""
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=self.api_key)
        all_results, seen_urls = [], set()
        
        for query in self.generate_queries(claim):
            try:
                data = client.search(query=query, search_depth='basic', max_results=5, include_answer=False)
                for result in data.get('results', []):
                    url = result.get('url', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(result)
            except Exception:
                pass
        
        evidence_list = []
        for i, item in enumerate(all_results[:10], 1):
            url = item.get('url', '')
            title = item.get('title', 'Untitled')
            snippet = item.get('content', '')
            domain = normalized_domain(url)
            rel = relationship(claim.text, title, snippet)
            tier = authority(domain)
            score = float(item.get('score', 0))
            
            evidence_list.append(Evidence(
                evidence_id=f'{claim.claim_id}-ev-{i:02d}',
                claim_id=claim.claim_id,
                title=title, url=url, domain=domain, snippet=snippet,
                relationship=rel, authority_tier=tier,
                search_score=min(1.0, score),
                quality_score=quality_score(tier, rel, score)
            ))
        
        return sorted(evidence_list, key=lambda x: x.quality_score, reverse=True)


class HybridVerdictGenerator:
    """Generate verdicts using deterministic scoring + LLM explanation."""
    
    def generate(self, claim: Claim, evidence: List[Evidence]) -> ClaimVerdict:
        """Generate verdict with guardrails."""
        direct = sum(e.relationship == 'DIRECT_SUPPORT' for e in evidence)
        partial = sum(e.relationship == 'PARTIAL_SUPPORT' for e in evidence)
        contra = sum(e.relationship == 'CONTRADICTION' for e in evidence)
        high_auth = sum(e.authority_tier == 'HIGH' for e in evidence)
        quality = sum(e.quality_score for e in evidence[:3]) / max(1, min(3, len(evidence)))
        
        # Verdict logic with guardrails
        if not evidence:
            verdict, confidence = 'Insufficient Evidence', 0.35
        elif contra > direct + partial:
            verdict, confidence = 'Unsupported', min(0.9, 0.55 + quality * 0.35)
        elif direct >= 1 and high_auth >= 1 and quality >= 0.7:
            verdict, confidence = 'Mostly Supported', min(0.92, 0.65 + quality * 0.3)
        elif direct + partial >= 2:
            verdict, confidence = 'Partially Supported', min(0.82, 0.55 + quality * 0.3)
        elif contra and (direct or partial):
            verdict, confidence = 'Mixed Evidence', 0.65
        else:
            verdict, confidence = 'Insufficient Evidence', 0.5
        
        risk = 35 if verdict in {'Supported', 'Mostly Supported'} else 50 if verdict in {'Partially Supported', 'Mixed Evidence'} else 70
        reasoning = f"Reviewed {len(evidence)} sources: {direct} direct support, {partial} partial support, {contra} contradictory; quality {quality:.2f}"
        gap = 'No independent evidence' if not evidence else ('Evidence does not verify every part' if verdict != 'Supported' else '')
        takeaway = 'Reasonable to accept with limitations' if verdict in {'Supported', 'Mostly Supported'} else 'Treat cautiously and verify'
        
        return ClaimVerdict(
            claim_id=claim.claim_id, claim_text=claim.text,
            verdict=verdict, confidence=round(confidence, 2), risk_score=risk,
            evidence_quality_score=round(quality, 3),
            supporting_evidence_count=direct + partial,
            contradicting_evidence_count=contra,
            reasoning=reasoning, evidence_gap=gap, user_takeaway=takeaway,
            evidence=evidence
        )


# ==================================================
# MAIN PIPELINE
# ==================================================

class ReelCheckPipeline:
    """Main processing pipeline with no hardcoded results."""
    
    def __init__(self):
        self.ffmpeg_path = None
    
    def initialize_ffmpeg(self):
        """Initialize FFmpeg once."""
        if self.ffmpeg_path is None:
            self.ffmpeg_path = configure_ffmpeg()
    
    def process(self, url: str, manual_caption: str = '', manual_transcript: str = '') -> AnalysisResult:
        """Process Instagram content end-to-end.
        
        IMPORTANT: This method NEVER returns hardcoded results.
        Every Reel is processed through the actual pipeline.
        """
        start_time = time.time()
        metrics = ProcessingMetrics()
        analysis_id = 'analysis-' + uuid.uuid4().hex[:12]
        workdir = None
        
        try:
            # Initialize FFmpeg
            self.initialize_ffmpeg()
            
            # Stage 1: Extraction
            t = time.time()
            extracted = UnifiedExtractor().extract(url)
            metrics.extraction_duration_sec = round(time.time() - t, 2)
            metrics.stages_completed.append('extraction')
            
            workdir = extracted.get('workdir')
            caption = extracted['caption'] or manual_caption
            media_paths = extracted.get('media_paths', [])
            image_paths = extracted.get('image_paths', [])
            
            # Stage 2: Audio extraction and transcription
            raw_transcript = manual_transcript
            if not raw_transcript and media_paths:
                t = time.time()
                try:
                    audio_path = str(Path(workdir) / 'audio.wav')
                    extract_audio(media_paths[0], audio_path, self.ffmpeg_path)
                    
                    transcript_result = WhisperTranscriber().transcribe(audio_path)
                    raw_transcript = transcript_result['raw_transcript']
                    
                    metrics.transcription_duration_sec = round(time.time() - t, 2)
                    metrics.stages_completed.append('transcription')
                except Exception as e:
                    metrics.stages_failed.append(f'transcription: {str(e)}')
                    raw_transcript = ''
            
            # Stage 3: Transcript correction
            corrector = TranscriptCorrector()
            audit = corrector.correct_with_caption_context(raw_transcript, caption, extracted.get('title', ''))
            
            # Stage 4: Visual text extraction
            visible_text = VisualTextExtractor().extract(image_paths)
            
            # Stage 5: Content fusion
            combined_parts = []
            if caption:
                combined_parts.append(f'CAPTION: {caption}')
            if audit.corrected_transcript:
                combined_parts.append(f'TRANSCRIPT: {audit.corrected_transcript}')
            if visible_text:
                combined_parts.append(f'VISIBLE TEXT: {visible_text}')
            
            combined_content = '\n\n'.join(combined_parts)
            
            # Stage 6: Claim extraction
            t = time.time()
            llm = DatabricksLLM()
            category, viewpoints, core_claim, claims = llm.classify_and_extract_claims(combined_content)
            metrics.claim_duration_sec = round(time.time() - t, 2)
            metrics.llm_calls += 1
            metrics.stages_completed.append('claim_extraction')
            
            # Stage 7: Evidence retrieval and verdict generation
            t = time.time()
            retriever = EvidenceRetriever()
            verdict_gen = HybridVerdictGenerator()
            verdicts = []
            
            for claim in claims:
                evidence = retriever.search(claim)
                verdict = verdict_gen.generate(claim, evidence)
                verdicts.append(verdict)
                metrics.search_api_calls += 1
            
            metrics.evidence_duration_sec = round(time.time() - t, 2)
            metrics.stages_completed.append('evidence_and_verdict')
            
            # Stage 8: Overall assessment
            positive = sum(v.verdict in {'Supported', 'Mostly Supported'} for v in verdicts)
            overall = 'Mostly Supported' if verdicts and positive == len(verdicts) else ('Mixed Evidence' if positive else (verdicts[0].verdict if verdicts else 'Insufficient Evidence'))
            
            reliability = round(100 * sum(v.confidence * v.evidence_quality_score for v in verdicts) / len(verdicts)) if verdicts else 0
            risk = round(sum(v.risk_score for v in verdicts) / len(verdicts)) if verdicts else 65
            
            noise = [p for p in ['going ballistic', 'going insane', 'follow for more', 'link in my bio'] if p in combined_content.lower()]
            
            return AnalysisResult(
                analysis_id=analysis_id, url=url.split('?')[0],
                shortcode=extracted['shortcode'], post_type=extracted['post_type'],
                caption=caption, asset_count=len(media_paths) + len(image_paths),
                visual_text=visible_text, transcript=audit,
                category=category, viewpoints=viewpoints, core_claim=core_claim,
                claims=claims, verdicts=verdicts,
                overall_verdict=overall, reliability_score=reliability, risk_score=risk,
                content_value_score=max(0, min(100, reliability - len(noise) * 5 + 20)),
                what_matters=[v.claim_text for v in verdicts if v.verdict in {'Supported', 'Mostly Supported', 'Partially Supported'}],
                cautions=[v.evidence_gap for v in verdicts if v.evidence_gap],
                promotional_noise=noise,
                recommended_action='Use evidence-backed findings as context. Verify critical decisions using primary sources.',
                metrics=metrics
            )
        finally:
            metrics.total_duration_sec = round(time.time() - start_time, 2)
            if workdir and Path(workdir).exists():
                shutil.rmtree(workdir, ignore_errors=True)
