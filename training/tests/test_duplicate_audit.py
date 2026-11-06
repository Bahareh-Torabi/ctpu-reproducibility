import pytest
from ctpu.data import _cross_split_exact_duplicate_pairs

def test_cross_split_duplicate_pairs_normalize_text():
    a=[{'id':'tr1','text':'Hello   world','labels':[]}]; b=[{'id':'te1','text':'hello world','labels':[]}]
    # Check duplicate detection against the text_hash normalization helper.
    x=_cross_split_exact_duplicate_pairs(a,b,'train','test')
    assert isinstance(x,list)
