import tempfile,unittest
from pathlib import Path
from resume_holdout import SAMPLER,merge_rows,partition,sha,validate_bindings

class ResumeContracts(unittest.TestCase):
    def test_partial_chat_is_replayed_whole_and_merge_is_exact(self):
        chats=[dict(id='a',positions=[2,5]),dict(id='b',positions=[1,4])]
        old=[dict(id='a',position=2,value='old'),dict(id='a',position=5,value='old'),dict(id='b',position=1,value='discard')]
        new=[dict(id='b',position=1,value='new'),dict(id='b',position=4,value='new')]
        self.assertEqual(partition(chats,old)[0],[chats[1]])
        self.assertEqual([r['value'] for r in merge_rows(chats,old,new)],['old','old','new','new'])
        with self.assertRaises(ValueError):merge_rows(chats,old,new[:1])
        with self.assertRaises(ValueError):merge_rows(chats,old,new+new[:1])
        with self.assertRaises(ValueError):merge_rows(chats,old,new+[old[0]])
    def test_changed_runtime_or_sampler_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'native';path.write_bytes(b'original')
            contract=dict(sampler=SAMPLER,sha256=dict(native=sha(path)))
            validate_bindings(contract,dict(native=path))
            path.write_bytes(b'different')
            with self.assertRaises(ValueError):validate_bindings(contract,dict(native=path))
            with self.assertRaises(ValueError):validate_bindings(dict(contract,sampler={}),dict(native=path))
if __name__=='__main__':unittest.main()
