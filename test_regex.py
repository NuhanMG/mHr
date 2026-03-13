import re

# Simulate what the LLM output would look like - single backslashes on Windows
text1 = "Download from C:\\Users\\Nuhan\\Videos\\HR\\data\\forms\\Salary Advance Form.pdf and submit."
text2 = "Policy on Foreign Travel (located at: C:\\Users\\Nuhan\\Videos\\HR\\data\\policies\\HR 0028.3 Policy on Foreign Travel.pdf)"
text3 = "Just a filename: Salary Advance Form.pdf"
text4 = "See C:\\Users\\Nuhan\\Videos\\HR\\data\\forms\\Salary Advance Form.pdf and also C:\\Users\\Nuhan\\Videos\\HR\\data\\policies\\HR 0028.3 Policy on Foreign Travel.pdf for details."

# The regex pattern as written in backend.py
pat_path = r'(?:(?:[A-Za-z]:\\|/)[\w\\/.:\- ]+?[\\/])([^\\/\n]+\.pdf)'
pat_located = r'\(?\s*(?:located|found|available|stored)\s+at:\s*([^)\n]+\.pdf)\s*\)?'

for i, text in enumerate([text1, text2, text3, text4], 1):
    print(f"\n--- Test {i} ---")
    print(f"Input:  {text}")
    result = re.sub(pat_path, r'\1', text)
    result = re.sub(pat_located, r'(\1)', result)
    print(f"Output: {result}")

    # Debug: show matches
    matches = re.findall(pat_path, text)
    print(f"Path matches: {matches}")
