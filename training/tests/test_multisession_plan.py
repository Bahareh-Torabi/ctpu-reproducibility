from ctpu.multisession import campaign_plan, expected_campaign_run_count


def test_multisession_campaign_counts_and_membership():
    assert expected_campaign_run_count('reuters21578') == 43
    assert expected_campaign_run_count('aapd') == 43
    assert expected_campaign_run_count('eurlex57k') == 21

    r = campaign_plan('reuters21578')
    a = campaign_plan('aapd')
    e = campaign_plan('eurlex57k')

    assert ('ctpu', 20260805) in r
    assert ('pairwise', 20260922) in r
    assert ('param_pcgrad', 20260829) in r
    assert ('param_pcgrad', 20260910) not in r

    assert ('ctpu', 20260922) in a
    assert ('param_pcgrad', 20260829) in a

    assert ('ctpu', 20260829) in e
    assert ('ctpu', 20260910) not in e
    assert all(m != 'score_auxproj' for m, _ in e)
    assert all(m != 'param_pcgrad' for m, _ in e)
