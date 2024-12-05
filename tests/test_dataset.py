import sys
import os
import pandas as pd
import transformers
# Get the relevant directories
project_root = os.getcwd()
src_path = os.path.join(project_root, 'src')

# Add src directory to sys.path if not already there
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import my custom modules
import dataset  # noqa: E402


##############################
# Helper functions
##############################
def get_relpath(dir):
    project_root = os.getcwd()
    return os.path.relpath(dir, project_root)


##############################
# Tests
##############################
class TestDatasetUtils:
    def test_create_training_pairs(self):
        """
        Tests the `create_training_pairs` function.
        """
        # Create dummy sentences of known authorship
        dummy_A_texts = ["dummy A text {}".format(i) for i in range(2)]
        # Create dummy sentences of unknown authorship
        dummy_notA_texts = ["dummy Imposter text {}".format(i) for i in range(2)]
        result = dataset.create_training_pairs(dummy_A_texts, dummy_notA_texts)
        expected = pd.DataFrame(
            [('dummy A text 0', 'dummy A text 1', 1),
             ('dummy A text 1', 'dummy Imposter text 1', 0)],
            columns=['Sentence A', 'Sentence B', 'Label']
        )

        assert len(result) == 2
        assert result.iloc[0]["Sentence A"] == expected.iloc[0]["Sentence A"]
        assert result.iloc[0]["Sentence B"] == expected.iloc[0]["Sentence B"]
        assert result.iloc[0]["Label"] == expected.iloc[0]["Label"]
        assert result.iloc[1]["Sentence A"].startswith("dummy A text ")
        assert result.iloc[1]["Sentence A"][13] in ["0", "1"]
        assert result.iloc[1]["Sentence B"].startswith("dummy Imposter text ")
        assert result.iloc[1]["Sentence B"][20] in ["0", "1"]
        assert result.iloc[1]["Label"] == 0


class TestCustomDataset:
    """
    Tests the CustomDataset class.
    """
    @classmethod
    def setup_class(cls):
        cls.dataset = dataset.CustomDataset('data/test') 

    def test_directory_structure(self):
        assert get_relpath(self.dataset._data_dir) == 'data/test'
        assert get_relpath(self.dataset._A_dir) == 'data/test/A'
        assert get_relpath(self.dataset._notA_dir) == 'data/test/notA'
        assert get_relpath(self.dataset._U_dir) == 'data/test/U'

    def test_text_loading(self):
        assert self.dataset._A_text == 'foo bar'
        assert self.dataset._notA_text == 'baz qux'
        assert self.dataset._U_text == 'quux quuz'

    def test_tokenization(self):
        tokens = self.dataset._A_tokens
        assert isinstance(
            tokens,
            transformers.tokenization_utils_base.BatchEncoding
        )
        assert all(
            k in tokens
            for k in ['input_ids', 'token_type_ids', 'attention_mask']
        )