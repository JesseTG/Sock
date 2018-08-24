try
    (fromjson | .text, .full_text | values | gsub("\n"; " <newline> "))
catch
    empty
