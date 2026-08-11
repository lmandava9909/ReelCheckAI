from transcript_correction import TranscriptCorrector

def test_reference_correction():
    audit=TranscriptCorrector().correct_with_caption_context('The United States is looking to Ben Chinese AI equipment.','THE U.S. BANNED CHINESE AI EQUIPMENT!')
    assert 'ban chinese ai equipment' in audit.corrected_transcript.lower()
