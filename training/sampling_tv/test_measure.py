import importlib.util
import math
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('measure',Path(__file__).with_name('measure.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
class MetricContract(unittest.TestCase):
    def test_shifted_mass_is_two_percent_not_top1_flip(self):
        p=[[0,.51],[1,.49]]; q=[[0,.49],[1,.51]]
        self.assertAlmostEqual(m.tv(p,q),.02)
        self.assertEqual(m.tv(p,p),0)
        self.assertEqual(m.tv([[0,1]],[[1,1]]),1)
        with self.assertRaises(ValueError): m.tv([[0,.5]],q)
        with self.assertRaises(ValueError): m.tv([[0,.5],[0,.5]],q)
    def test_truncation_is_exact_post_sampler_zero(self):
        x=[math.log(.51),math.log(.49)]+[-1000.]*22
        p=m.distribution(x); self.assertEqual(len(p),2); self.assertAlmostEqual(p[0][1],.51)
        self.assertEqual(m.distribution([math.log(.96),math.log(.04)]+[-1000.]*22),[[0,1.]])
        self.assertEqual([i for i,p in m.distribution([0.]*24)],list(range(19)))
        with self.assertRaises(ValueError): m.distribution([float('nan')]*24)
    def test_only_aligned_generation_positions_are_counted(self):
        import tempfile, json, struct, types, hashlib
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); ids=[10,11,12,0,1]
            (root/'capture.complete').write_text('fixture')
            (root/'metadata.json').write_text('{}')
            (root/'requests.jsonl').write_text(json.dumps(dict(id='a',token_ids=ids,prompt_tokens=3,generated_tokens=2,vocab_size=24))+'\n')
            (root/'answers.jsonl').write_text(json.dumps(dict(id='a',bucket='text',pieces_hex=['61','62'],stop='eog'))+'\n')
            teacher=[]; student=[]
            for off in range(2):
                prefix=ids[:3+off]
                teacher.append(dict(id='a',offset=off,phase='assistant_generation',bucket='text',sampled_token=ids[3+off],prefix_sha256=hashlib.sha256(struct.pack('<'+'I'*len(prefix),*prefix)).hexdigest(),distribution=[[0,.51],[1,.49]]))
                student.append(dict(id='a',offset=off,distribution=[[0,.49],[1,.51]]))
            def write(name,rows): (root/name).write_text(''.join(json.dumps(x)+'\n' for x in rows))
            write('teacher.jsonl',teacher); write('student.jsonl',student)
            m.compare(types.SimpleNamespace(output=str(root)))
            result=json.loads((root/'result.json').read_text())
            self.assertEqual(result['overall']['tokens'],2)
            self.assertAlmostEqual(result['overall']['mean_percent'],2)
            (root/'result.json').unlink(); (root/'positions.jsonl').unlink()
            teacher[0]['phase']='prompt_processing'; write('teacher.jsonl',teacher)
            with self.assertRaises(ValueError): m.compare(types.SimpleNamespace(output=str(root)))
if __name__=='__main__': unittest.main()
