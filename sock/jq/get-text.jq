try
    (fromjson | .text, .full_text | values | gsub("\n"; " <newline> "))
    # Try to get either the .text or the .full_text field from this JSON object,
    # then tokenize its \n's as <newline>'s...
catch
    # ...but if that fails (e.g. invalid JSON), throw out this object
    empty
