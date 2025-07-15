from .wordsegall_txt import TCfenci_all

class wordsegall_txt:
    @staticmethod
    def TCfenci_all(raw_path, resultpath, max_seq_length=128, eval_batch_size=3):
        return TCfenci_all(raw_path, resultpath, max_seq_length, eval_batch_size)

__all__ = ['wordsegall_txt']