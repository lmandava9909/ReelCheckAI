from models import Claim

def test_claim_schema():
    claim=Claim(claim_id='c1',text='The agency announced a new public policy.',risk_domain='Policy')
    assert claim.risk_domain=='Policy'
