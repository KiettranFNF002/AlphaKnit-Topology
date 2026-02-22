import os

test_dir = "tests"
for f in os.listdir(test_dir):
    if f.endswith(".py"):
        path = os.path.join(test_dir, f)
        try:
            with open(path, "rb") as rb:
                content = rb.read()
            # Try decoding as utf-8
            content.decode("utf-8")
            print(f"✅ {f}: UTF-8 OK")
        except UnicodeDecodeError:
            print(f"❌ {f}: Encoding Error! Attempting fix...")
            # Try common fallback encodings
            found = False
            for enc in ["latin-1", "windows-1252"]:
                try:
                    text = content.decode(enc)
                    # Rewrite as UTF-8
                    with open(path, "w", encoding="utf-8") as wf:
                        wf.write(text)
                    print(f"✨ {f}: Fixed (Converted from {enc} to UTF-8)")
                    found = True
                    break
                except UnicodeDecodeError:
                    continue
            if not found:
                print(f"💀 {f}: Could not fix automatically.")
