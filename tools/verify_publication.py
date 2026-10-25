"""Verify the curated publication manifest without Torch or a GPU."""
import hashlib
from pathlib import Path

def main():
 root=Path(__file__).resolve().parents[1]
 lines=(root/'PUBLICATION.sha256').read_text().splitlines()
 for line in lines:
  expected,name=line.split('  ',1)
  path=(root/name).resolve()
  if root not in path.parents:raise ValueError('manifest path outside checkout')
  assert path.is_file(),name
  assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
 print('PUBLICATION_HASHES_PASS',len(lines))

if __name__=='__main__':main()
