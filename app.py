import html
import streamlit as st
from processor import ReelCheckPipeline

st.set_page_config(page_title='ReelCheck AI',page_icon='✓',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>
.stApp{background:linear-gradient(180deg,#fff 0,#f5f5f7 60%);color:#1d1d1f;font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue",Arial,sans-serif}.block-container{max-width:1180px;padding-top:1.4rem}.hero{text-align:center;padding:62px 18px 34px}.eyebrow{color:#0071e3;font-weight:700}.hero h1{font-size:clamp(44px,7vw,76px);letter-spacing:-.055em;margin:.2em 0}.hero p{font-size:21px;line-height:1.5;color:#6e6e73;max-width:780px;margin:auto}.glass{background:rgba(255,255,255,.85);border:1px solid rgba(0,0,0,.07);border-radius:28px;padding:26px;box-shadow:0 18px 60px rgba(0,0,0,.07);backdrop-filter:blur(18px)}.metric{background:#fff;border:1px solid rgba(0,0,0,.07);border-radius:22px;padding:20px;min-height:122px}.metric b{font-size:29px}.metric small{color:#6e6e73}.claim{background:#fff;border:1px solid rgba(0,0,0,.07);border-radius:22px;padding:22px;margin:14px 0}.pill{display:inline-block;padding:6px 10px;border-radius:999px;background:#eef6ff;color:#0366c7;font-weight:700;font-size:12px;margin:3px}
</style>''',unsafe_allow_html=True)
st.markdown('<section class="hero"><div class="eyebrow">Independent evidence for social claims</div><h1>Know what a post is really claiming.</h1><p>ReelCheck extracts captions, spoken audio, and visible text; identifies verifiable claims; checks independent sources; and explains what is supported, uncertain, or misleading.</p></section>',unsafe_allow_html=True)
if 'result' not in st.session_state: st.session_state.result=None
st.markdown('<div class="glass">',unsafe_allow_html=True)
url=st.text_input('Instagram URL',placeholder='https://www.instagram.com/reel/...')
with st.expander('If Instagram blocks automated extraction'):
    manual_caption=st.text_area('Paste the caption (optional)')
    manual_transcript=st.text_area('Paste a transcript (optional)')
clicked=st.button('Analyze post',type='primary',use_container_width=True)
st.caption('Public links work best. Temporary media is deleted after analysis. Results are informational and not financial, legal, medical, or safety advice.')
st.markdown('</div>',unsafe_allow_html=True)
if clicked:
    with st.status('Analyzing post…',expanded=True) as status:
        try:
            for label in ['Reading post','Extracting media and visible text','Checking transcript','Identifying claims','Finding evidence','Building result']: st.write(label)
            st.session_state.result=ReelCheckPipeline().process(url,manual_caption,manual_transcript)
            status.update(label='Analysis complete',state='complete',expanded=False)
        except Exception as e:
            status.update(label='Analysis needs attention',state='error'); st.error('The post could not be fully analyzed. Try the caption or transcript fallback.'); st.caption(str(e))
r=st.session_state.result
if r:
    st.divider(); st.subheader('ReelCheck Brief')
    cols=st.columns(4)
    data=[('Overall assessment',r.overall_verdict,'Evidence-grounded conclusion'),('Reliability',f'{r.reliability_score}/100','Claim and evidence confidence'),('Risk',f'{r.risk_score}/100','Potential harm if relied upon'),('Transcript trust',f'{round(r.transcript.quality_score*100)}%','Extraction and correction quality')]
    for col,(label,val,caption) in zip(cols,data): col.markdown(f'<div class="metric"><small>{label}</small><br><b>{val}</b><br><small>{caption}</small></div>',unsafe_allow_html=True)
    tabs=st.tabs(['Overview','Claims','Evidence','Trust Audit','Recommended Action'])
    with tabs[0]:
        st.markdown('### Primary claim'); st.write(r.core_claim or 'No clear core factual claim was identified.')
        st.markdown('### What the post gets right')
        for item in r.what_matters or ['No evidence-backed claim was established.']: st.success(item)
        st.markdown('### What requires caution')
        for item in r.cautions or r.promotional_noise or ['No additional caution was generated.']: st.warning(item)
        with st.expander('Extracted content'):
            st.write('**Caption**'); st.write(r.caption or 'Unavailable'); st.write('**Transcript**'); st.write(r.transcript.corrected_transcript or 'Unavailable')
            if r.visual_text: st.write('**Visible text**'); st.write(r.visual_text)
    with tabs[1]:
        if not r.verdicts: st.info('No sufficiently specific verifiable claims were identified.')
        for v in r.verdicts:
            st.markdown(f'<div class="claim"><span class="pill">{html.escape(v.verdict)}</span><span class="pill">Confidence {round(v.confidence*100)}%</span><h3>{html.escape(v.claim_text)}</h3><p>{html.escape(v.reasoning)}</p><p><b>User takeaway:</b> {html.escape(v.user_takeaway)}</p></div>',unsafe_allow_html=True)
    with tabs[2]:
        for v in r.verdicts:
            st.markdown(f'### {v.claim_text}')
            for e in v.evidence:
                st.markdown(f'**{e.title}**'); st.caption(f'{e.domain} · {e.authority_tier} authority · {e.relationship} · quality {e.quality_score:.2f}'); st.write(e.snippet[:500]); st.link_button('Open source',e.url)
    with tabs[3]:
        st.metric('Transcript quality',f'{round(r.transcript.quality_score*100)}%'); st.write(f'Words analyzed: {r.transcript.word_count}'); st.write(f'Corrections applied: {len(r.transcript.corrections)}')
        for c in r.transcript.corrections: st.info(f'“{c.original_phrase}” → “{c.corrected_phrase}” · {round(c.confidence*100)}% · {c.rationale}')
        for flag in r.transcript.review_flags: st.warning(flag)
        with st.expander('Compare raw and corrected transcript'): st.write('**Raw**'); st.write(r.transcript.raw_transcript); st.write('**Corrected**'); st.write(r.transcript.corrected_transcript)
    with tabs[4]:
        st.markdown('### Recommended action'); st.write(r.recommended_action)
        st.markdown('### Before acting'); st.write('Review the strongest primary or high-authority source. Treat weak or related context as background, not proof. Verify financial, health, legal, and safety decisions independently.')
