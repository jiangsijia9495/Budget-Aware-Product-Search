from budget_aware_search.text import build_product_text, clean_text


def test_clean_text_removes_html_and_whitespace():
    assert clean_text("  <b>Red</b>   shoe &amp; sock ") == "Red shoe & sock"


def test_build_product_text_uses_labeled_fields():
    row = {
        "product_title": "Trail Shoe",
        "product_brand": "Acme",
        "product_color": "Blue",
        "product_description": None,
    }
    text = build_product_text(
        row,
        ["product_title", "product_brand", "product_color", "product_description"],
    )
    assert text == "title: Trail Shoe [SEP] brand: Acme [SEP] color: Blue"
