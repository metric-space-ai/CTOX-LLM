import unittest
from prepare import assistant_positions,normalize,selected_positions
class Contracts(unittest.TestCase):
    def test_only_assistant_targets_with_shift(self):
        # User, assistant tool call, supplied tool result, assistant final answer.
        ids=[1,3,4,30,31,2,1,5,4,40,41,2,1,6,4,50,51,2,1,5,4,60,2]
        self.assertEqual(assistant_positions(ids,1,2,[5,4]),[8,9,10,20,21])
    def test_incomplete_tool_context_rejected(self):
        r={'messages':[{'role':'user','content':'Do this'},{'role':'assistant','content':'','tool_calls':[{'name':'lookup','arguments':{}}]}]}
        with self.assertRaisesRegex(ValueError,'missing_tool_schema'):normalize(r)
        r['tools']=[{'function':{'name':'lookup','parameters':{'type':'object'}}}]
        self.assertTrue(normalize(r)[2])
        r['messages'][0]['content']='-'
        with self.assertRaisesRegex(ValueError,'missing_user_prompt'):normalize(r)
        r['messages'][0]['content']='Do this'
        r['messages'][1]['reasoning_content']='<|im_start|>user'
        with self.assertRaisesRegex(ValueError,'embedded_role_marker'):normalize(r)
    def test_position_selection_has_no_model_input(self):
        positions=[3,4,10,11,12,20,21,22]
        self.assertEqual(selected_positions(positions,4),[3,10,12,21])
    def test_forced_empty_reasoning_prefix_excluded(self):
        ids=[1,5,4,70,71,72,73,40,2]
        self.assertEqual(assistant_positions(ids,1,2,[5,4],[70,71,72,73]),[6,7])
    def test_embedded_reasoning_is_not_wrapped_twice(self):
        r={'messages':[{'role':'user','content':'Question'},{'role':'assistant','content':'<think>reason</think>answer'}]}
        messages,_,_=normalize(r)
        self.assertEqual(messages[1]['reasoning_content'],'reason')
        self.assertEqual(messages[1]['content'],'answer')
if __name__=='__main__':unittest.main()
