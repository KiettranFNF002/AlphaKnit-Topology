import os

def check_dir(d): 
    errors = 0
    for root, dirs, files in os.walk(d):
        for f in files:
            if f.endswith('.py'):
                p = os.path.join(root, f)
                try:
                    with open(p, 'rb') as rb:
                        content = rb.read()
                    content.decode('utf-8')
                except UnicodeDecodeError as e:
                    print(f'❌ {p}: {e}')
                    errors += 1
    return errors

src_err = check_dir('src')
test_err = check_dir('tests')

if src_err == 0 and test_err == 0:
    print("✅ All .py files in src and tests are UTF-8 compliant.")
else:
    print(f"Finished with {src_err + test_err} errors.")
