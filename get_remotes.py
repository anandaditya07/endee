import subprocess
try:
    result = subprocess.check_output(["git", "remote", "-v"], stderr=subprocess.STDOUT)
    print(result.decode())
except Exception as e:
    print(str(e))
