"""Run all available-data experiments plus the revised H4 estimand analysis.

python -B revision_20261028_run_experiments.py --list
python -B revision_20261028_run_experiments.py --tag reproduce_20261029

Writes new tagged outputs only. Does not reconstruct the unavailable original
MSAF training pipeline. The underlying stages have been executed separately;
the combined launcher is an orchestration convenience.
"""
from pathlib import Path
import argparse, ast, datetime, subprocess, sys
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag',default='reproduce_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    ap.add_argument('--list',action='store_true')
    a=ap.parse_args()
    if a.list:
        subprocess.run([sys.executable,'-B',str(R/'audit_20261028_rerun_all_v2.py'),'--list'],check=True)
        print('h4_estimands revision_20261028_h4.py')
        return
    if not a.tag.replace('_','').replace('-','').isalnum():raise ValueError('Use an alphanumeric tag')
    if any(R.glob(a.tag+'*')):raise FileExistsError('Choose a new tag; existing files are retained.')
    subprocess.run([sys.executable,'-B',str(R/'audit_20261028_rerun_all_v2.py'),'--tag',a.tag],cwd=R,check=True)
    src=R/'revision_20261028_h4.py'
    class Prefix(ast.NodeTransformer):
        def visit_Constant(self,node):
            if isinstance(node.value,str):node.value=node.value.replace('audit_20261028',a.tag).replace('revision_20261028',a.tag)
            return node
    tree=Prefix().visit(ast.parse(src.read_text(encoding='utf-8-sig'),filename=str(src)));ast.fix_missing_locations(tree)
    sys.argv=[str(src)]
    exec(compile(tree,str(src),'exec'),{'__name__':'__main__','__file__':str(src)})
    print('Completed all available-data computational stages:',a.tag)
if __name__=='__main__':main()
