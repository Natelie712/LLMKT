import torch
import torch.utils.data as data
import numpy as np

# Total questions available across all courses in the curriculum
TOTAL_Q_SLOTS = 305


class getReader:
    """
    Simple reader for ReKT-style sequence text files.

    Expected format (three lines per user sequence):
        line 1: sequence length L (int)
        line 2: comma-separated question ids (length L)
        line 3: comma-separated 0/1 answers (length L)
    """

    def __init__(self, path: str):
        self.path = path

    def readData(self):
        problem_list = []
        ans_list = []
        split_char = ","

        with open(self.path, "r", encoding="utf-8") as read:
            cur_len = None
            cur_probs = None

            for index, line in enumerate(read):
                mod = index % 3

                if mod == 0:
                    # sequence length (can be ignored, we will recompute)
                    try:
                        cur_len = int(line.strip())
                    except ValueError:
                        cur_len = None

                elif mod == 1:
                    problems = line.strip().split(split_char) if line.strip() else []
                    # convert to int
                    problems = list(map(int, problems)) if problems else []
                    cur_probs = problems

                elif mod == 2:
                    ans = line.strip().split(split_char) if line.strip() else []
                    ans = list(map(float, ans)) if ans else []
                    ans = [int(x) for x in ans]

                    # sanity: length should match
                    if cur_probs is not None and len(cur_probs) == len(ans):
                        problem_list.append(cur_probs)
                        ans_list.append(ans)

        return problem_list, ans_list


class KT_Dataset(data.Dataset):
    """
    Dataset used by ReKT training / evaluation.

    Given:
        - problem_list: list of question-id sequences
        - skill_list:   list of concept-id sequences (same shape as problem_list)
        - ans_list:     list of 0/1 correctness sequences

    We:
        - drop sequences shorter than min_problem_num
        - split sequences longer than max_problem_num into multiple
          non-overlapping chunks of length max_problem_num
        - right-align and zero-pad each chunk to length `max_problem_num`
        - then create (last, next) pairs of length (max_problem_num - 1)
          along with a mask indicating valid positions
    """

    def __init__(self, problem_max, problem_list, skill_list, ans_list,
                 min_problem_num, max_problem_num):
        super().__init__()
        self.problem_max = problem_max
        self.min_problem_num = min_problem_num
        self.max_problem_num = max_problem_num

        self.problem_list = []
        self.ans_list = []
        self.skill_list = []
        self.completion_rates = []  # Track completion rate for each chunk
        self.user_ids = []  # Track user ID for each chunk

        for user_id, (problem, ans, skill) in enumerate(zip(problem_list, ans_list, skill_list)):
            assert len(problem) == len(ans) == len(skill)
            num = len(problem)

            # drop too-short sequences
            if num < self.min_problem_num:
                continue

            # Calculate completion rate for this student (before chunking)
            # num = total questions this student answered
            completion_rate = num / float(TOTAL_Q_SLOTS)

            # split long sequences into chunks of max_problem_num
            start = 0
            while start < num:
                end = min(start + self.max_problem_num, num)
                sub_problem = problem[start:end]
                sub_ans = ans[start:end]
                sub_skill = skill[start:end]
                sub_len = end - start

                if sub_len >= self.min_problem_num:
                    self.problem_list.append(sub_problem)
                    self.ans_list.append(sub_ans)
                    self.skill_list.append(sub_skill)
                    # All chunks from same student get same completion rate and user_id
                    self.completion_rates.append(completion_rate)
                    self.user_ids.append(user_id)

                start = end

    def __len__(self):
        return len(self.problem_list)

    def __getitem__(self, index):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        now_problem = self.problem_list[index]
        now_skill = self.skill_list[index]
        now_ans = self.ans_list[index]

        num = len(now_problem)
        # safety check
        if num > self.max_problem_num:
            now_problem = now_problem[-self.max_problem_num:]
            now_skill = now_skill[-self.max_problem_num:]
            now_ans = now_ans[-self.max_problem_num:]
            num = self.max_problem_num

        # right-align and zero-pad to length max_problem_num
        use_problem = np.zeros(self.max_problem_num, dtype=int)
        use_skill = np.zeros(self.max_problem_num, dtype=int)
        use_ans = np.zeros(self.max_problem_num, dtype=float)
        use_mask = np.zeros(self.max_problem_num, dtype=int)

        use_problem[-num:] = np.array(now_problem, dtype=int)
        use_skill[-num:] = np.array(now_skill, dtype=int)
        use_ans[-num:] = np.array(now_ans, dtype=float)
        use_mask[-num:] = 1

        # build last/next sequences (length max_problem_num - 1)
        last_problem = use_problem[:-1]
        next_problem = use_problem[1:]

        last_skill = use_skill[:-1]
        next_skill = use_skill[1:]

        last_ans = use_ans[:-1]
        next_ans = use_ans[1:]

        # mask for transitions: valid where both last and next are real
        mask = np.zeros(self.max_problem_num - 1, dtype=int)
        # if original sequence length is num, then we have (num - 1) valid transitions
        if num > 1:
            mask[-(num - 1):] = 1

        # convert to tensors and push to device
        last_problem = torch.from_numpy(last_problem).to(device).long()
        next_problem = torch.from_numpy(next_problem).to(device).long()

        last_ans = torch.from_numpy(last_ans).to(device).long()
        next_ans = torch.from_numpy(next_ans).to(device).float()

        last_skill = torch.from_numpy(last_skill).to(device).long()
        next_skill = torch.from_numpy(next_skill).to(device).long()

        mask_tensor = torch.tensor(mask == 1).to(device)
        
        # Get completion rate and user_id for this sequence
        completion_rate = self.completion_rates[index]
        user_id = self.user_ids[index]

        return last_problem, last_skill, last_ans, next_problem, next_skill, next_ans, mask_tensor, completion_rate, user_id


def getLoader(problem_max, pro_path, skill_path, batch_size,
              is_train, min_problem_num, max_problem_num):
    """
    Factory used by run.py.

    Parameters
    ----------
    problem_max : int
        Maximum number of distinct problems (unused here but kept
        for interface compatibility).
    pro_path, skill_path : str
        Paths to *.txt files produced by preprocessing. Each must
        follow the three-line ReKT format.
    batch_size : int
    is_train : bool
        If True, DataLoader is shuffled.
    min_problem_num, max_problem_num : int

    Returns
    -------
    torch.utils.data.DataLoader
        Yields batches of:
          (last_problem, last_skill, last_ans,
           next_problem, next_skill, next_ans,
           mask)
    """
    problem_list, ans_list = getReader(pro_path).readData()
    skill_list, _ = getReader(skill_path).readData()

    dataset = KT_Dataset(
        problem_max,
        problem_list,
        skill_list,
        ans_list,
        min_problem_num,
        max_problem_num,
    )
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return loader
