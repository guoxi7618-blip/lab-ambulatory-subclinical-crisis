"""Versioned launcher for all computational audit stages; never overwrites outputs.

python -B audit_20261028_rerun_all.py --list
python -B audit_20261028_rerun_all.py --tag rerun_20261029

To retain historical scripts exactly, this launcher changes only the string
literal output-name prefix in memory, using Python AST (not on-disk editing).
Input CSV paths, equations, seeds and source scripts remain unchanged. Each
stage executes in a separate Python process and fails visibly on errors.
"""
from pathlib import Path
import argparse,ast,datetime,subprocess,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
STAGES=[
 ('integrity','audit_20261028_integrity_v2.py'),
 ('integrity_addendum','audit_20261028_integrity_addendum.py'),
 ('h1','audit_20261028_h1.py'),
 ('h1_definitions','audit_20261028_h1_followup.py'),
 ('models','audit_20261028_models.py'),
 ('model_diagnostics','audit_20261028_refinement.py'),
 ('daily_definitions','audit_20261028_daily_refinement.py'),
 ('supplementary','audit_20261028_supplementary_audit_v2.py'),
 ('calibration_sensitivity','audit_20261028_calibration_sensitivity.py'),
]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--tag',default='rerun_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));ap.add_argument('--list',action='store_true');ap.add_argument('--stage',choices=[s[0] for s in STAGES],help=argparse.SUPPRESS);a=ap.parse_args()
 if not a.tag.replace('_','').replace('-','').isalnum():raise ValueError('Tag must contain only letters, digits, underscore or hyphen')
 if a.list:
  for name,file in STAGES:print(name,file)
  return
 if a.stage:
  source=ROOT/dict(STAGES)[a.stage]
  class Prefix(ast.NodeTransformer):
   def visit_Constant(self,node):
    if isinstance(node.value,str):node.value=node.value.replace('audit_20261028',a.tag)
    return node
  tree=Prefix().visit(ast.parse(source.read_text(encoding='utf-8-sig'),filename=str(source)));ast.fix_missing_locations(tree)
  sys.argv=[str(source)]
  scope={'__name__':'__main__','__file__':str(source),'__builtins__':__builtins__}
  exec(compile(tree,str(source),'exec'),scope)
  return
 if any(ROOT.glob(a.tag+'*')):raise FileExistsError('This tag has existing outputs; choose a new --tag. Nothing was deleted.')
 for name,file in STAGES:
  if not (ROOT/file).is_file():raise FileNotFoundError(file)
 for name,file in STAGES:
  print('RUN',name,flush=True)
  subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--tag',a.tag,'--stage',name],cwd=ROOT,check=True)
 print('Completed available computational stages:',a.tag,flush=True)
 print('MSAF raw-signal preprocessing/training remains unavailable without missing original inputs.')
if __name__=='__main__':main()
