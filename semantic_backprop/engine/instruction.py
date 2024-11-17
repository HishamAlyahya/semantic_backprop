from typing import Optional

class Instruction:
    candiate_instructions = [        
        'Devise an experiment to help solve that problem.',
        'Make a list of ideas for solving this problem.',
        'Measure progress on this problem.',
        'What are the key assumptions underlying this problem?',
        'What are the potential risks and drawbacks of each solution?',
        'What are the alternative perspectives or viewpoints on this problem?',
        'Break down this problem into smaller, more manageable parts.',
    ]
    thinking_styles = [
        'Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.',
        'Try critical Thinking. This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.'
    ]

    def __init__(self, instruciton_str: Optional[str]=None):
        if instruciton_str is None:
            instruciton_str = 'Work out an intermediate step that helps solve the problem.'

        self.instruciton_str = instruciton_str

